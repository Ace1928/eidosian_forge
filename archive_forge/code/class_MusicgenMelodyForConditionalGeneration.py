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
@add_start_docstrings('The composite Musicgen Melody model with a text and audio conditional models, a MusicgenMelody decoder and an audio encoder, for music generation tasks with one or both of text and audio prompts.', MUSICGEN_MELODY_START_DOCSTRING, '\n        text_encoder (`Optional[PreTrainedModel]`, *optional*): Text encoder.\n        audio_encoder (`Optional[PreTrainedModel]`, *optional*): Audio code decoder.\n        decoder (`Optional[MusicgenMelodyForCausalLM]`, *optional*): MusicGen Melody decoder used to generate audio codes.\n    ')
class MusicgenMelodyForConditionalGeneration(PreTrainedModel):
    config_class = MusicgenMelodyConfig
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: MusicgenMelodyConfig=None, text_encoder: Optional[PreTrainedModel]=None, audio_encoder: Optional[PreTrainedModel]=None, decoder: Optional[MusicgenMelodyForCausalLM]=None):
        if config is None and None in (text_encoder, audio_encoder, decoder):
            raise ValueError('Either a configuration has to be provided, or all three of text encoder, audio encoder and Musicgen Melody decoder.')
        if config is None:
            config = MusicgenMelodyConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config)
        elif not isinstance(config, self.config_class):
            raise ValueError(f'Config: {config} has to be of type {self.config_class}')
        super().__init__(config)
        if text_encoder is None:
            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)
        if audio_encoder is None:
            audio_encoder = AutoModel.from_config(config.audio_encoder)
        if decoder is None:
            decoder = MusicgenMelodyForCausalLM(config.decoder)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder
        self.text_encoder.config = self.config.text_encoder
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder
        if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size:
            self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)
        if self.config.num_chroma != self.decoder.config.hidden_size:
            self.audio_enc_to_dec_proj = nn.Linear(self.config.num_chroma, self.decoder.config.hidden_size)
        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(f'The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head')
        self.post_init()

    def _init_weights(self, module):
        std = self.decoder.config.initializer_factor
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def tie_weights(self):
        if self.config.tie_encoder_decoder:
            decoder_base_model_prefix = self.decoder.base_model_prefix
            tied_weights = self._tie_encoder_decoder_weights(self.text_encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix, 'text_encoder')
            self._dynamic_tied_weights_keys = tied_weights

    def get_text_encoder(self):
        return self.text_encoder

    def get_encoder(self):
        return self.get_text_encoder()

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_sub_models_pretrained(cls, text_encoder_pretrained_model_name_or_path: str=None, audio_encoder_pretrained_model_name_or_path: str=None, decoder_pretrained_model_name_or_path: str=None, *model_args, **kwargs) -> PreTrainedModel:
        """
        Instantiate a text encoder, an audio encoder, and a MusicGen decoder from one, two or three base classes of the
        library from pretrained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            text_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the audio encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text encoder configuration, use the prefix *text_encoder_* for each configuration
                  parameter.
                - To update the audio encoder configuration, use the prefix *audio_encoder_* for each configuration
                  parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import MusicgenMelodyForConditionalGeneration

        >>> # initialize a musicgen model from a t5 text encoder, encodec audio encoder, and musicgen decoder
        >>> model = MusicgenMelodyForConditionalGeneration.from_sub_models_pretrained(
        ...     text_encoder_pretrained_model_name_or_path="google-t5/t5-base",
        ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",
        ...     decoder_pretrained_model_name_or_path="facebook/musicgen-melody",
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./musicgen-ft")
        >>> # load fine-tuned model
        >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("./musicgen-ft")
        ```"""
        kwargs_text_encoder = {argument[len('text_encoder_'):]: value for argument, value in kwargs.items() if argument.startswith('text_encoder_')}
        kwargs_audio_encoder = {argument[len('audio_encoder_'):]: value for argument, value in kwargs.items() if argument.startswith('audio_encoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in kwargs.items() if argument.startswith('decoder_')}
        for key in kwargs_text_encoder.keys():
            del kwargs['text_encoder_' + key]
        for key in kwargs_audio_encoder.keys():
            del kwargs['audio_encoder_' + key]
        for key in kwargs_decoder.keys():
            del kwargs['decoder_' + key]
        text_encoder = kwargs_text_encoder.pop('model', None)
        if text_encoder is None:
            if text_encoder_pretrained_model_name_or_path is None:
                raise ValueError('If `text_encoder_model` is not defined as an argument, a `text_encoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_text_encoder:
                encoder_config, kwargs_text_encoder = AutoConfig.from_pretrained(text_encoder_pretrained_model_name_or_path, **kwargs_text_encoder, return_unused_kwargs=True)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {text_encoder_pretrained_model_name_or_path} as a text_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_text_encoder['config'] = encoder_config
            text_encoder = AutoModel.from_pretrained(text_encoder_pretrained_model_name_or_path, *model_args, **kwargs_text_encoder)
        audio_encoder = kwargs_audio_encoder.pop('model', None)
        if audio_encoder is None:
            if audio_encoder_pretrained_model_name_or_path is None:
                raise ValueError('If `audio_encoder_model` is not defined as an argument, an `audio_encoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_audio_encoder:
                encoder_config, kwargs_audio_encoder = AutoConfig.from_pretrained(audio_encoder_pretrained_model_name_or_path, **kwargs_audio_encoder, return_unused_kwargs=True)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {audio_encoder_pretrained_model_name_or_path} as an audio_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_audio_encoder['config'] = encoder_config
            audio_encoder = AutoModel.from_pretrained(audio_encoder_pretrained_model_name_or_path, *model_args, **kwargs_audio_encoder)
        decoder = kwargs_decoder.pop('model', None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError('If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True)
                if isinstance(decoder_config, MusicgenMelodyConfig):
                    decoder_config = decoder_config.decoder
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers.")
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True
                kwargs_decoder['config'] = decoder_config
            if kwargs_decoder['config'].is_decoder is False or kwargs_decoder['config'].add_cross_attention is False:
                logger.warning(f'Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_sub_models_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_sub_models_pretrained(...)`')
            decoder = MusicgenMelodyForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        config = MusicgenMelodyConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config, **kwargs)
        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder, config=config)

    @add_start_docstrings_to_model_forward(MUSICGEN_MELODY_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MusicgenMelodyOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.BoolTensor]=None, input_features: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, MusicgenMelodyOutputWithPast]:
        """
        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
        >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

        >>> inputs = processor(
        ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        ...     padding=True,
        ...     return_tensors="pt",
        ... )

        >>> pad_token_id = model.generation_config.pad_token_id
        >>> decoder_input_ids = (
        ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
        ...     * pad_token_id
        ... )

        >>> logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
        >>> logits.shape  # (bsz * num_codebooks, encoder_len + tgt_len, vocab_size)
        torch.Size([8, 249, 2048])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_text_encoder = {argument[len('text_encoder_')]: value for argument, value in kwargs.items() if argument.startswith('text_encoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in kwargs.items() if argument.startswith('decoder_')}
        if encoder_hidden_states is None:
            if inputs_embeds is not None or input_ids is not None:
                encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs_text_encoder)
                encoder_hidden_states = encoder_outputs[0]
                if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size:
                    encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
            if attention_mask is not None and encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]
            if encoder_hidden_states is not None and input_features is None:
                input_features = torch.zeros((encoder_hidden_states.shape[0], 1, self.config.num_chroma), device=self.device, dtype=self.dtype)
                input_features[:, :, 0] = 1
            if input_features is not None:
                audio_hidden_states = input_features
                if self.config.num_chroma != self.decoder.config.hidden_size:
                    audio_hidden_states = self.audio_enc_to_dec_proj(audio_hidden_states)
                if audio_hidden_states.shape[1] < self.config.chroma_length:
                    n_repeat = int(math.ceil(self.config.chroma_length / audio_hidden_states.shape[1]))
                    audio_hidden_states = audio_hidden_states.repeat(1, n_repeat, 1)
                else:
                    logger.warning(f'The conditional audio signal is of length {audio_hidden_states.shape[1]}, which exceedsthe maximum chroma duration of {self.config.chroma_length}.The audio will be truncated to {self.config.chroma_length} frames.')
                audio_hidden_states = audio_hidden_states[:, :self.config.chroma_length]
                if encoder_hidden_states is not None:
                    encoder_hidden_states = torch.cat([audio_hidden_states, encoder_hidden_states], dim=1)
                else:
                    encoder_hidden_states = audio_hidden_states
        if labels is not None and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_hidden_states, inputs_embeds=decoder_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, past_key_values=past_key_values, return_dict=return_dict, **kwargs_decoder)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + (encoder_hidden_states,)
            else:
                return decoder_outputs + (encoder_hidden_states,)
        return MusicgenMelodyOutputWithPast(loss=loss, logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values, hidden_states=decoder_outputs.hidden_states, attentions=decoder_outputs.attentions, encoder_hidden_states=encoder_hidden_states)

    def prepare_inputs_for_generation(self, decoder_input_ids, encoder_hidden_states=None, past_key_values=None, attention_mask=None, decoder_attention_mask=None, decoder_head_mask=None, use_cache=None, decoder_delay_pattern_mask=None, guidance_scale=None, **kwargs):
        if decoder_delay_pattern_mask is None:
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(decoder_input_ids, self.generation_config.pad_token_id, max_length=self.generation_config.max_length)
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)
        if guidance_scale is not None and guidance_scale > 1:
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
            encoder_hidden_states = None
        return {'input_ids': None, 'encoder_hidden_states': encoder_hidden_states, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'decoder_head_mask': decoder_head_mask, 'use_cache': use_cache}

    def _prepare_decoder_input_ids_for_generation(self, batch_size: int, model_input_name: str, model_kwargs: Dict[str, torch.Tensor], decoder_start_token_id: int=None, bos_token_id: int=None, device: torch.device=None) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        if model_kwargs is not None and 'decoder_input_ids' in model_kwargs:
            decoder_input_ids = model_kwargs.pop('decoder_input_ids')
        elif 'input_ids' in model_kwargs and model_input_name != 'input_ids':
            decoder_input_ids = model_kwargs.pop('input_ids')
        else:
            decoder_input_ids = None
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device) * decoder_start_token_id
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if 'decoder_attention_mask' in model_kwargs:
                decoder_attention_mask = model_kwargs['decoder_attention_mask']
                decoder_attention_mask = torch.cat((torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask), dim=-1)
                model_kwargs['decoder_attention_mask'] = decoder_attention_mask
        return (decoder_input_ids, model_kwargs)

    def _prepare_encoder_hidden_states_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str]=None, guidance_scale: Optional[float]=None) -> Dict[str, Any]:
        encoder_hidden_states = None
        encoder_attention_mask = model_kwargs.pop('attention_mask')
        if inputs_tensor is not None:
            encoder = self.get_text_encoder()
            if hasattr(encoder, '_hf_hook'):
                encoder._hf_hook.io_same_device = True
            irrelevant_prefix = ['decoder_', 'use_cache']
            encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any((argument.startswith(p) for p in irrelevant_prefix))}
            encoder_signature = set(inspect.signature(encoder.forward).parameters)
            encoder_accepts_wildcard = 'kwargs' in encoder_signature or 'model_kwargs' in encoder_signature
            if not encoder_accepts_wildcard:
                encoder_kwargs = {argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature}
            model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
            encoder_kwargs['return_dict'] = True
            encoder_kwargs[model_input_name] = inputs_tensor
            if encoder_attention_mask is not None:
                encoder_kwargs['attention_mask'] = encoder_attention_mask
            encoder_hidden_states = encoder(**encoder_kwargs).last_hidden_state
            if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size:
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
            if guidance_scale is not None and guidance_scale > 1:
                encoder_hidden_states = torch.concatenate([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=0)
                if encoder_attention_mask is not None:
                    encoder_attention_mask = torch.concatenate([encoder_attention_mask, torch.zeros_like(encoder_attention_mask)], dim=0)
            if encoder_attention_mask is not None:
                encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None]
        audio_hidden_states = model_kwargs.get('input_features', None)
        if inputs_tensor is not None:
            if audio_hidden_states is not None:
                null_audio_hidden_states = torch.zeros_like(audio_hidden_states)
            else:
                null_audio_hidden_states = torch.zeros((inputs_tensor.shape[0], 1, self.config.num_chroma), device=self.device, dtype=self.dtype)
            null_audio_hidden_states[:, :, 0] = 1
            if audio_hidden_states is None:
                audio_hidden_states = null_audio_hidden_states
        if audio_hidden_states is not None:
            if guidance_scale is not None and guidance_scale > 1:
                audio_hidden_states = torch.concatenate([audio_hidden_states, null_audio_hidden_states], dim=0)
            if self.config.num_chroma != self.decoder.config.hidden_size:
                audio_hidden_states = self.audio_enc_to_dec_proj(audio_hidden_states)
            if audio_hidden_states.shape[1] < self.config.chroma_length:
                n_repeat = int(math.ceil(self.config.chroma_length / audio_hidden_states.shape[1]))
                audio_hidden_states = audio_hidden_states.repeat(1, n_repeat, 1)
            audio_hidden_states = audio_hidden_states[:, :self.config.chroma_length]
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.cat([audio_hidden_states, encoder_hidden_states], dim=1)
            else:
                encoder_hidden_states = audio_hidden_states
        model_kwargs['encoder_hidden_states'] = encoder_hidden_states
        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError('Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or model.decoder.resize_token_embeddings(...))')

    def _maybe_initialize_input_ids_for_generation(self, inputs: Optional[torch.Tensor]=None, bos_token_id: Optional[int]=None, model_kwargs: Optional[Dict[str, torch.Tensor]]=None) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs
        if bos_token_id is None:
            raise ValueError('`bos_token_id` has to be defined when no `input_ids` are provided.')
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

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
            synced_gpus (`bool`, *optional*):
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

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
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
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        model_kwargs['output_attentions'] = generation_config.output_attentions
        model_kwargs['output_hidden_states'] = generation_config.output_hidden_states
        model_kwargs['use_cache'] = generation_config.use_cache
        model_kwargs['guidance_scale'] = generation_config.guidance_scale
        if model_kwargs.get('attention_mask', None) is None:
            model_kwargs['attention_mask'] = self._prepare_attention_mask_for_generation(inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id)
        if 'encoder_hidden_states' not in model_kwargs:
            model_kwargs = self._prepare_encoder_hidden_states_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, guidance_scale=generation_config.guidance_scale)
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs, decoder_start_token_id=generation_config.decoder_start_token_id, bos_token_id=generation_config.bos_token_id, device=inputs_tensor.device)
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            logger.warning(f'Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.')
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(={generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(f'Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than the maximum length ({generation_config.max_length})')
        if input_ids_seq_length >= generation_config.max_length:
            logger.warning(f'Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.')
        input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(input_ids, pad_token_id=generation_config.decoder_start_token_id, max_length=generation_config.max_length)
        model_kwargs['decoder_delay_pattern_mask'] = decoder_delay_pattern_mask
        if streamer is not None:
            streamer.put(input_ids.cpu())
        is_greedy_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is False)
        is_sample_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is True)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None
        logits_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=None, logits_processor=logits_processor)
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1 when doing greedy search, but is {generation_config.num_return_sequences}.')
            outputs = self.greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        elif is_sample_gen_mode:
            logits_warper = self._get_logits_warper(generation_config)
            input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            outputs = self.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        else:
            raise ValueError('Got incompatible mode for generation, should be one of greedy or sampling. Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`.')
        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs['decoder_delay_pattern_mask'])
        output_ids = output_ids[output_ids != generation_config.pad_token_id].reshape(batch_size, self.decoder.num_codebooks, -1)
        output_ids = output_ids[None, ...]
        audio_scales = model_kwargs.get('audio_scales')
        if audio_scales is None:
            audio_scales = [None] * batch_size
        if self.decoder.config.audio_channels == 1:
            output_values = self.audio_encoder.decode(output_ids, audio_scales=audio_scales).audio_values
        else:
            codec_outputs_left = self.audio_encoder.decode(output_ids[:, :, ::2, :], audio_scales=audio_scales)
            output_values_left = codec_outputs_left.audio_values
            codec_outputs_right = self.audio_encoder.decode(output_ids[:, :, 1::2, :], audio_scales=audio_scales)
            output_values_right = codec_outputs_right.audio_values
            output_values = torch.cat([output_values_left, output_values_right], dim=1)
        if generation_config.return_dict_in_generate:
            outputs.sequences = output_values
            return outputs
        else:
            return output_values

    def _update_model_kwargs_for_generation(self, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool=False, standardize_cache_format: bool=False, model_inputs: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        model_kwargs['past_key_values'] = self._extract_past_from_model_output(outputs, standardize_cache_format=standardize_cache_format)
        if getattr(outputs, 'state', None) is not None:
            model_kwargs['state'] = outputs.state
        if 'token_type_ids' in model_kwargs:
            token_type_ids = model_kwargs['token_type_ids']
            model_kwargs['token_type_ids'] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
        if 'decoder_attention_mask' in model_kwargs:
            decoder_attention_mask = model_kwargs['decoder_attention_mask']
            model_kwargs['decoder_attention_mask'] = torch.cat([decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))], dim=-1)
        return model_kwargs