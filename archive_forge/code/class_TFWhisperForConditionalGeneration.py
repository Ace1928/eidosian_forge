from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
@add_start_docstrings('The Whisper Model with a language modeling head. Can be used for automatic speech recognition.', WHISPER_START_DOCSTRING)
class TFWhisperForConditionalGeneration(TFWhisperPreTrainedModel, TFCausalLanguageModelingLoss):
    base_model_prefix = 'model'
    _keys_to_ignore_on_load_missing = ['encoder.version', 'decoder.version', 'proj_out.weight']
    _keys_to_ignore_on_save = ['proj_out.weight']

    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = TFWhisperMainLayer(config, name='model')

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int) -> keras.layers.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_features: TFModelInputType | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, decoder_position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, decoder_inputs_embeds: Optional[Tuple[Union[np.ndarray, tf.Tensor]]]=None, labels: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple[tf.Tensor], TFSeq2SeqLMOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoProcessor, TFWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="tf")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(input_features=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        outputs = self.model(input_features, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        decoder_last_hidden_state = outputs[0]
        lm_logits = tf.matmul(decoder_last_hidden_state, self.get_output_embeddings().weights, transpose_b=True)
        loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSeq2SeqLMOutput(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def generate(self, inputs: Optional[tf.Tensor]=None, generation_config: Optional[GenerationConfig]=None, logits_processor: Optional[TFLogitsProcessorList]=None, seed: Optional[List[int]]=None, return_timestamps: Optional[bool]=None, task: Optional[str]=None, language: Optional[str]=None, is_multilingual: Optional[bool]=None, prompt_ids: Optional[tf.Tensor]=None, return_token_timestamps=None, **kwargs):
        """
        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate, e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`tf.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If unset the method
                initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs` should of in
                the format of `input_ids`. For encoder-decoder models *inputs* can represent any of `input_ids`,
                `input_values`, `input_features`, or `pixel_values`.
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
            seed (`List[int]`, *optional*):
                Random seed to control sampling, containing two integers, used when `do_sample` is `True`. See the
                `seed` argument from stateless functions in `tf.random`.
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `TFWhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`tf.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `tf.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a `tf.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.TFGreedySearchDecoderOnlyOutput`],
                    - [`~generation.TFSampleDecoderOnlyOutput`],
                    - [`~generation.TFBeamSearchDecoderOnlyOutput`],
                    - [`~generation.TFBeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.TFGreedySearchEncoderDecoderOutput`],
                    - [`~generation.TFSampleEncoderDecoderOutput`],
                    - [`~generation.TFBeamSearchEncoderDecoderOutput`],
                    - [`~generation.TFBeamSampleEncoderDecoderOutput`]

        """
        if generation_config is None:
            generation_config = self.generation_config
        if return_timestamps is not None:
            if not hasattr(generation_config, 'no_timestamps_token_id'):
                raise ValueError('You are trying to return timestamps, but the generation config is not properly set. Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363')
            generation_config.return_timestamps = return_timestamps
        else:
            generation_config.return_timestamps = False
        if language is not None:
            language = language.lower()
            generation_config.language = language
        if task is not None:
            generation_config.task = task
        forced_decoder_ids = None
        if hasattr(self.config, 'forced_decoder_ids') and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif hasattr(self.generation_config, 'forced_decoder_ids') and self.generation_config.forced_decoder_ids is not None:
            forced_decoder_ids = self.generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.get('forced_decoder_ids', None)
        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, 'language'):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f'<|{TO_LANGUAGE_CODE[generation_config.language]}|>'
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f'<|{generation_config.language}|>'
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(f'Unsupported language: {generation_config.language}. Language should be one of: {(list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys()))}.')
                if language_token not in generation_config.lang_to_id:
                    raise ValueError(f'{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`.(You should just add it to the generation config)')
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))
            if hasattr(generation_config, 'task'):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(f'The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`')
            elif hasattr(generation_config, 'task_to_id'):
                forced_decoder_ids.append((2, generation_config.task_to_id['transcribe']))
            if hasattr(generation_config, 'no_timestamps_token_id') and (not generation_config.return_timestamps):
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))
        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids
        if prompt_ids is not None:
            if kwargs.get('decoder_start_token_id') is not None:
                raise ValueError('When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten.')
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            text_prompt_ids = text_prompt_ids[-self.config.max_length // 2 - 1:]
            kwargs.update({'decoder_start_token_id': decoder_start_token_id})
            specified_max_length = kwargs.pop('max_new_tokens', None) or kwargs.pop('max_length', None)
            default_max_length = generation_config.max_new_tokens or generation_config.max_length
            non_prompt_max_length = specified_max_length or default_max_length
            kwargs['max_new_tokens'] = non_prompt_max_length + len(text_prompt_ids)
            non_prompt_forced_decoder_ids = kwargs.pop('forced_decoder_ids', None) or generation_config.forced_decoder_ids
            forced_decoder_ids = [*text_prompt_ids, generation_config.decoder_start_token_id, *[token for _rank, token in non_prompt_forced_decoder_ids]]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids
        if generation_config.return_timestamps:
            raise ValueError("`TFWhisperForConditionalGeneration` doesn't support returning the timestamps yet.")
        if return_token_timestamps:
            kwargs['output_attentions'] = True
            kwargs['return_dict_in_generate'] = True
            if getattr(generation_config, 'task', None) == 'translate':
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, 'alignment_heads'):
                raise ValueError('Model generation config has no `alignment_heads`, token-level timestamps not available. See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config.')
        outputs = super().generate(inputs, generation_config, logits_processor, **kwargs)
        if return_token_timestamps and hasattr(generation_config, 'alignment_heads'):
            outputs['token_timestamps'] = self._extract_token_timestamps(outputs, generation_config.alignment_heads)
        return outputs

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        return TFSeq2SeqLMOutput(logits=output.logits, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, use_cache=None, encoder_outputs=None, attention_mask=None, decoder_attention_mask=None, **kwargs):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        if decoder_attention_mask is not None:
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        elif past_key_values is not None:
            decoder_position_ids = past_key_values[0][0].shape[2]
        else:
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])
        decoder_position_ids = tf.broadcast_to(decoder_position_ids, decoder_input_ids.shape)
        return {'input_features': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'use_cache': use_cache, 'decoder_attention_mask': decoder_attention_mask, 'decoder_position_ids': decoder_position_ids}

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'model', None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)