import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
@add_start_docstrings('The speech-to-text SeamlessM4T Model transformer which can be used for S2TT.', SEAMLESS_M4T_START_DOCSTRING)
class SeamlessM4TForSpeechToText(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['text_decoder', 't2u_model', 'vocoder']
    main_input_name = 'input_features'
    _tied_weights_keys = ['lm_head.weight', 'text_decoder.embed_tokens.weight']

    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        return self.speech_encoder

    def get_decoder(self):
        return self.text_decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_SPEECH_INPUTS_DOCSTRING)
    def forward(self, input_features: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.speech_encoder(input_features=input_features, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        encoder_attention_mask = attention_mask
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(encoder_outputs[0].device)
            encoder_attention_mask = _compute_new_attention_mask(hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths)
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def generate(self, input_features=None, tgt_lang=None, generation_config=None, logits_processor=None, stopping_criteria=None, prefix_allowed_tokens_fn=None, synced_gpus=False, **kwargs):
        """
        Generates sequences of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.

            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
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
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible
            [`~utils.ModelOutput`] types are:
                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        text_decoder_input_ids = kwargs.pop('decoder_input_ids', None)
        if tgt_lang is not None:
            inputs = kwargs.get('input_embeds') if input_features is None else input_features
            inputs = inputs if inputs is not None else kwargs.get('encoder_outputs', {'last_hidden_state': None})['last_hidden_state']
            batch_size = len(inputs)
            if hasattr(self.generation_config, 'text_decoder_lang_to_code_id'):
                tgt_lang = tgt_lang.replace('__', '')
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(f'`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in\n                        {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}')
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
            else:
                raise ValueError("This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps\n                    the target language to the right token id. Make sure to load the right generation config.")
        else:
            logger.warning('You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get\n                a correct generation, otherwise the generation will probably make no sense.')
        return super().generate(input_features, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, decoder_input_ids=text_decoder_input_ids, **kwargs)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past