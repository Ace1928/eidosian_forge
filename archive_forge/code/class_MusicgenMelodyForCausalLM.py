import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForTextEncoding
from .configuration_musicgen_melody import MusicgenMelodyConfig, MusicgenMelodyDecoderConfig
from ..deprecated._archive_maps import MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
@add_start_docstrings('The Musicgen Melody decoder model with a language modelling head on top.', MUSICGEN_MELODY_START_DOCSTRING)
class MusicgenMelodyForCausalLM(MusicgenMelodyPreTrainedModel):

    def __init__(self, config: MusicgenMelodyDecoderConfig):
        super().__init__(config)
        self.model = MusicgenMelodyModel(config)
        self.num_codebooks = config.num_codebooks
        self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)])
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @add_start_docstrings_to_model_forward(MUSICGEN_MELODY_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MusicgenMelodyOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple, MusicgenMelodyOutputWithPast]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not implemented for MusicgenMelody.')
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return MusicgenMelodyOutputWithPast(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, head_mask=None, past_key_values=None, use_cache=True, delay_pattern_mask=None, guidance_scale=None, **kwargs):
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(input_ids, pad_token_id=self.generation_config.pad_token_id, max_length=self.generation_config.max_length)
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        if guidance_scale is not None and guidance_scale > 1:
            input_ids = input_ids.repeat((2, 1))
            if attention_mask is not None:
                attention_mask = attention_mask.repeat((2, 1))
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concatenate([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=0)
            if encoder_attention_mask is not None:
                encoder_attention_mask = torch.concatenate(encoder_attention_mask, torch.zeros_like(encoder_attention_mask), dim=0)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            encoder_hidden_states = None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'encoder_hidden_states': encoder_hidden_states, 'encoder_attention_mask': encoder_attention_mask, 'head_mask': head_mask, 'past_key_values': past_key_values, 'use_cache': use_cache}

    def build_delay_pattern_mask(self, input_ids: torch.LongTensor, pad_token_id: int, max_length: int=None):
        """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [P, -1, -1, -1, -1, P, P, P]
        - [P, P, -1, -1, -1, -1, P, P]
        - [P, P, P, -1, -1, -1, -1, P]
        - [P, P, P, P, -1, -1, -1, -1]
        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [P, a, b, -1, -1, P, P, P]
        - [P, P, c, d, -1, -1, P, P]
        - [P, P, P, e, f, -1, -1, P]
        - [P, P, P, P, g, h, -1, -1]
        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
        input_ids = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
        bsz, num_codebooks, seq_len = input_ids.shape
        max_length = max_length if max_length is not None else self.generation_config.max_length
        input_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1
        channel_codebooks = num_codebooks // 2 if self.config.audio_channels == 2 else num_codebooks
        if max_length < 2 * channel_codebooks - 1:
            return (input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1))
        for codebook in range(channel_codebooks):
            if self.config.audio_channels == 1:
                input_ids_shifted[:, codebook, codebook:seq_len + codebook] = input_ids[:, codebook]
            else:
                input_ids_shifted[:, 2 * codebook, codebook:seq_len + codebook] = input_ids[:, 2 * codebook]
                input_ids_shifted[:, 2 * codebook + 1, codebook:seq_len + codebook] = input_ids[:, 2 * codebook + 1]
        delay_pattern = torch.triu(torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal=max_length - channel_codebooks + 1)
        delay_pattern = delay_pattern + torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))
        if self.config.audio_channels == 2:
            delay_pattern = delay_pattern.repeat_interleave(2, dim=0)
        mask = ~delay_pattern.to(input_ids.device)
        input_ids = mask * input_ids_shifted + ~mask * pad_token_id
        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            first_start_id = seq_len
        pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
        input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
        return (input_ids, pattern_mask)

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, synced_gpus: Optional[bool]=None, streamer: Optional['BaseStreamer']=None, **kwargs):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get('attention_mask', None) is None:
                logger.warning("The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.')
            generation_config.pad_token_id = eos_token_id
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = input_ids.shape[0] // self.num_codebooks
        model_kwargs['output_attentions'] = generation_config.output_attentions
        model_kwargs['output_hidden_states'] = generation_config.output_hidden_states
        model_kwargs['use_cache'] = generation_config.use_cache
        model_kwargs['guidance_scale'] = generation_config.guidance_scale
        if model_kwargs.get('attention_mask', None) is None:
            model_kwargs['attention_mask'] = self._prepare_attention_mask_for_generation(input_ids, generation_config.pad_token_id, generation_config.eos_token_id)
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and (generation_config.max_length == 20):
            logger.warning(f'Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation.')
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(={generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(f'Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than the maximum length ({generation_config.max_length})')
        if input_ids_seq_length >= generation_config.max_length:
            logger.warning(f'Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.')
        input_ids, delay_pattern_mask = self.build_delay_pattern_mask(input_ids, pad_token_id=generation_config.decoder_start_token_id, max_length=generation_config.max_length)
        if streamer is not None:
            streamer.put(input_ids.cpu())
        model_kwargs['delay_pattern_mask'] = delay_pattern_mask
        is_greedy_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is False)
        is_sample_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is True)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None
        logits_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=input_ids, prefix_allowed_tokens_fn=None, logits_processor=logits_processor)
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1 when doing greedy search, but is {generation_config.num_return_sequences}.')
            outputs = self._greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        elif is_sample_gen_mode:
            logits_warper = self._get_logits_warper(generation_config)
            input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, **model_kwargs)
            outputs = self._sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        else:
            raise ValueError('Got incompatible mode for generation, should be one of greedy or sampling. Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`.')
        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_ids = self.apply_delay_pattern_mask(output_ids, model_kwargs['delay_pattern_mask'])
        output_ids = output_ids[output_ids != generation_config.pad_token_id].reshape(batch_size, self.num_codebooks, -1)
        if generation_config.return_dict_in_generate:
            outputs.sequences = output_ids
            return outputs
        else:
            return output_ids