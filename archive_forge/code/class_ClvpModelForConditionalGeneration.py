import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
@add_start_docstrings('The composite CLVP model with a text encoder, speech encoder and speech decoder model.The speech decoder model generates the speech_ids from the text and the text encoder and speech encoder workstogether to filter out the best speech_ids.', CLVP_START_DOCSTRING)
class ClvpModelForConditionalGeneration(ClvpPreTrainedModel):
    config_class = ClvpConfig

    def __init__(self, config: ClvpConfig):
        super().__init__(config)
        if not isinstance(config.text_config, ClvpEncoderConfig):
            raise ValueError(f'config.text_config is expected to be of type `ClvpEncoderConfig` but is of type {type(config.text_config)}.')
        if not isinstance(config.speech_config, ClvpEncoderConfig):
            raise ValueError(f'config.speech_config is expected to be of type `ClvpEncoderConfig` but is of type {type(config.speech_config)}.')
        if not isinstance(config.decoder_config, ClvpDecoderConfig):
            raise ValueError(f'config.decoder_config is expected to be of type `ClvpDecoderConfig` but is of type {type(config.decoder_config)}.')
        self.conditioning_encoder = ClvpConditioningEncoder(config)
        self.speech_decoder_model = ClvpForCausalLM(config.decoder_config)
        self.text_encoder_model = ClvpEncoder(config.text_config)
        self.speech_encoder_model = ClvpEncoder(config.speech_config)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.post_init()

    def fix_speech_decoder_output(self, speech_ids: torch.LongTensor) -> torch.LongTensor:
        """
        This method modifies the output of the decoder model, such as replacing the `eos_token_id` and changing the
        last few tokens of each sequence.

        Args:
            speech_ids (`torch.LongTensor`):
                This refers to the output of the decoder model.
        """
        decoder_fixing_codes = self.config.decoder_config.decoder_fixing_codes
        speech_ids = speech_ids[:, 1:]
        stop_token_indices = torch.where(speech_ids == self.speech_decoder_model.config.eos_token_id, 1, 0)
        speech_ids = torch.masked_fill(speech_ids, mask=stop_token_indices.bool(), value=decoder_fixing_codes[0])
        for i, each_seq_stop_token_index in enumerate(stop_token_indices):
            if each_seq_stop_token_index.sum() == 0:
                continue
            stm = each_seq_stop_token_index.argmax()
            speech_ids[i, stm:] = decoder_fixing_codes[0]
            if stm - 3 < speech_ids.shape[1]:
                speech_ids[i, -3:] = torch.tensor([decoder_fixing_codes[1:]], device=speech_ids.device, dtype=torch.long)
        return speech_ids

    def get_text_features(self, input_ids: Optional[torch.LongTensor]=None, text_encoder_inputs_embeds: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        """
        This method can be used to extract text_embeds from a text. The text embeddings obtained by applying the
        projection layer to the pooled output of the CLVP text encoder model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                [What are input IDs?](../glossary#input-ids)
            text_encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
                inputs_embeds for the text encoder model passed in place of `input_ids`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

        Returns:
            `torch.FloatTensor` of shape `(batch_size, output_dim)`:
                The text embeddings obtained by applying the projection layer to the pooled output of the CLVP Text
                Model.

        Examples:

        ```python
        >>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

        >>> # Define the Text
        >>> text = "This is an example text."

        >>> # Define processor and model
        >>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

        >>> # Generate processor output and text embeds
        >>> processor_output = processor(text=text, return_tensors="pt")
        >>> text_embeds = model.get_text_features(input_ids=processor_output["input_ids"])
        ```
        """
        outputs = self.text_encoder_model(input_ids=input_ids, inputs_embeds=text_encoder_inputs_embeds, attention_mask=attention_mask)
        return outputs[0]

    def get_speech_features(self, speech_ids: Optional[torch.LongTensor]=None, input_ids: Optional[torch.LongTensor]=None, input_features: Optional[torch.FloatTensor]=None, conditioning_encoder_inputs_embeds: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, **kwargs) -> torch.FloatTensor:
        """
        This method can be used to extract speech_embeds. The speech embeddings are obtained by applying the speech
        model on speech_ids. If speech_ids is not present but both input_ids and input_features are given then the
        decoder model will be used to first generate the speech_ids and then applying the speech model.

        Args:
            speech_ids (`torch.LongTensor` of shape `(batch_size, num_speech_ids)`, *optional*):
                Speech Tokens. Padding will be ignored by default should you provide it. If speech_ids are provided
                then input_ids and input_features will be automatically ignored.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Input text Tokens. Processed from the [`ClvpTokenizer`]. If speech_ids is not provided, then input_ids
                and input_features will be used.
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`, *optional*):
                Indicates log-melspectrogram representations for audio returned by [`ClvpFeatureExtractor`]. If
                speech_ids is not provided, then input_ids and input_features will be used.
            conditioning_encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
                inputs_embeds for `ClvpConditioningEncoder`. Can be used in place of `input_ids`.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding speech token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            generation_config (`GenerationConfig`, *optional*):
                generation config to control the generation of speech_ids if they are not provided.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, output_dim)`:
                The speech embeddings obtained by applying the projection layer to the pooled output of the CLVP Speech
                Model.

        Examples:

        ```python
        >>> import datasets
        >>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

        >>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
        >>> text = "This is an example text."
        >>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
        >>> _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()

        >>> # Define processor and model
        >>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

        >>> # Generate processor output and model output
        >>> processor_output = processor(raw_speech=audio, sampling_rate=sr, text=text, return_tensors="pt")
        >>> speech_embeds = model.get_speech_features(
        ...     input_ids=processor_output["input_ids"], input_features=processor_output["input_features"]
        ... )
        ```
        """
        if speech_ids is None:
            if input_ids is None and conditioning_encoder_inputs_embeds is None or input_features is None:
                raise ValueError('Either speech_ids or input_ids/conditioning_encoder_inputs_embeds and input_features must be provided.')
            if generation_config is None:
                generation_config = self.generation_config
            generation_config.update(**kwargs)
            conditioning_embeds = self.conditioning_encoder(input_features=input_features, input_ids=input_ids, inputs_embeds=conditioning_encoder_inputs_embeds, attention_mask=attention_mask)
            speech_ids = self.speech_decoder_model.generate(conditioning_embeds=conditioning_embeds, generation_config=generation_config)
            speech_ids = self.fix_speech_decoder_output(speech_ids[0])
        outputs = self.speech_encoder_model(input_ids=speech_ids, attention_mask=attention_mask)
        return outputs[0]

    @add_start_docstrings_to_model_forward(CLVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClvpOutput, config_class=ClvpConfig)
    def forward(self, input_ids: torch.LongTensor=None, input_features: torch.FloatTensor=None, conditioning_encoder_inputs_embeds: Optional[torch.FloatTensor]=None, text_encoder_inputs_embeds: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, return_loss: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=False, return_dict: Optional[bool]=None) -> Union[Tuple, ClvpOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> import datasets
        >>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

        >>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
        >>> text = "This is an example text."

        >>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
        >>> _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()

        >>> # Define processor and model
        >>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

        >>> # processor outputs and model outputs
        >>> processor_output = processor(raw_speech=audio, sampling_rate=sr, text=text, return_tensors="pt")
        >>> outputs = model(
        ...     input_ids=processor_output["input_ids"],
        ...     input_features=processor_output["input_features"],
        ...     return_dict=True,
        ... )
        ```
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        conditioning_embeds = self.conditioning_encoder(input_features=input_features, input_ids=input_ids, inputs_embeds=conditioning_encoder_inputs_embeds, attention_mask=attention_mask)
        decoder_outputs = self.speech_decoder_model(inputs_embeds=conditioning_embeds, output_hidden_states=output_hidden_states, return_dict=return_dict)
        speech_ids = decoder_outputs[0]
        if speech_ids.ndim == 3:
            speech_ids = speech_ids.argmax(2)
        speech_ids = self.fix_speech_decoder_output(speech_ids)
        speech_outputs = self.speech_encoder_model(input_ids=speech_ids, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_encoder_model(input_ids=input_ids, inputs_embeds=text_encoder_inputs_embeds, attention_mask=attention_mask, output_hidden_states=output_hidden_states, return_dict=return_dict)
        speech_embeds = speech_outputs[0]
        text_embeds = text_outputs[0]
        speech_embeds = speech_embeds / speech_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, speech_embeds.t()) * logit_scale
        logits_per_speech = logits_per_text.t()
        loss = None
        if return_loss:
            loss = clvp_loss(logits_per_text)
        if not return_dict:
            output = (logits_per_speech, logits_per_text, text_embeds, speech_embeds, text_outputs[2], speech_outputs[2])
            if output_hidden_states:
                output += (decoder_outputs[-1], text_outputs[-1], speech_outputs[-1])
            return (loss,) + output if loss is not None else output
        return ClvpOutput(loss=loss, logits_per_speech=logits_per_speech, logits_per_text=logits_per_text, text_embeds=text_embeds, speech_embeds=speech_embeds, text_model_output=text_outputs[2], speech_model_output=speech_outputs[2], decoder_hidden_states=decoder_outputs.hidden_states, text_encoder_hidden_states=text_outputs.hidden_states, speech_encoder_hidden_states=speech_outputs.hidden_states)

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor=None, input_features: torch.FloatTensor=None, attention_mask: Optional[torch.LongTensor]=None, generation_config: Optional[GenerationConfig]=None, pad_to_max_mel_tokens: Optional[int]=None, output_hidden_states: Optional[bool]=None, **kwargs):
        """
        Generate method for `ClvpModelForConditionalGeneration`, this method calls the `generate` method of
        `ClvpForCausalLM` and then uses those generated `speech_ids` to process `text_embeds` and `speech_embeds` using
        `ClvpEncoder`.

        Args:
            input_ids (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Input text Tokens. Processed from the [`ClvpTokenizer`].
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`, *optional*):
                Indicates log-melspectrogram representations for audio returned by [`ClvpFeatureExtractor`].
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding text token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            pad_to_max_mel_tokens (`int`, *optional*):
                Pads generated speech_ids to the specified value. This is to implement the same logic from the official
                repo, link: https://github.com/neonbjb/tortoise-tts/blob/80f89987a5abda5e2b082618cd74f9c7411141dc/tortoise/api.py#L430
                and to make sure the logits are same.
                This does not affect generation quality so please don't consider using it since it is less efficient.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of decoder model, text encoder and speech encoder models.

        Returns:
            `ClvpOutput` or tuple: A `ClvpOutput` (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a tuple.
        """
        sequence_length = input_ids.shape[-1]
        if sequence_length > self.config.decoder_config.max_text_tokens - 3:
            raise ValueError(f'Maximum sequence length reached! Found input_ids of length {sequence_length}.Please make sure that the maximum length of input_ids is {self.config.decoder_config.max_text_tokens - 3}')
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())
        input_ids, attention_mask = _pad_extra_bos_eos_tokens(input_ids, attention_mask, add_bos_token=False, bos_token_id=self.config.text_config.bos_token_id, eos_token_id=self.config.text_config.eos_token_id)
        conditioning_embeds = self.conditioning_encoder(input_features=input_features, input_ids=input_ids, attention_mask=attention_mask)
        decoder_outputs = self.speech_decoder_model.generate(conditioning_embeds=conditioning_embeds, generation_config=generation_config, output_hidden_states=output_hidden_states, return_dict=generation_config.return_dict_in_generate)
        if isinstance(decoder_outputs, ModelOutput):
            speech_ids = decoder_outputs.sequences
        if pad_to_max_mel_tokens is not None:
            padding_needed = pad_to_max_mel_tokens - speech_ids.shape[-1]
            speech_ids = torch.nn.functional.pad(speech_ids, (0, padding_needed), value=self.generation_config.eos_token_id)
        speech_ids = self.fix_speech_decoder_output(speech_ids)
        speech_outputs = self.speech_encoder_model(input_ids=speech_ids, output_hidden_states=output_hidden_states, return_dict=generation_config.return_dict_in_generate)
        text_outputs = self.text_encoder_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states, return_dict=generation_config.return_dict_in_generate)
        speech_embeds = speech_outputs[0]
        text_embeds = text_outputs[0]
        speech_embeds = speech_embeds / speech_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, speech_embeds.t()) * logit_scale
        logits_per_speech = logits_per_text.t()
        if not generation_config.return_dict_in_generate:
            output = (speech_ids, logits_per_speech, logits_per_text, text_embeds, speech_embeds, text_outputs[2], speech_outputs[2])
            if output_hidden_states:
                output += (decoder_outputs[-1], text_outputs[-1], speech_outputs[-1])
            return output
        return ClvpOutput(speech_ids=speech_ids, logits_per_speech=logits_per_speech, logits_per_text=logits_per_text, text_embeds=text_embeds, speech_embeds=speech_embeds, text_model_output=text_outputs[2], speech_model_output=speech_outputs[2], decoder_hidden_states=decoder_outputs.hidden_states, text_encoder_hidden_states=text_outputs.hidden_states, speech_encoder_hidden_states=speech_outputs.hidden_states)