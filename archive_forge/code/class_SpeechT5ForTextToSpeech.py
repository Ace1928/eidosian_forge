import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
@add_start_docstrings('SpeechT5 Model with a text encoder and a speech decoder.', SPEECHT5_START_DOCSTRING)
class SpeechT5ForTextToSpeech(SpeechT5PreTrainedModel):
    main_input_name = 'input_ids'

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        if config.vocab_size is None:
            raise ValueError(f"You are trying to instantiate {self.__class__} with a configuration that does not define the vocabulary size of the language model head. Please instantiate the model as follows: `SpeechT5ForTextToSpeech.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of your model's configuration.")
        text_encoder = SpeechT5EncoderWithTextPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        self.speecht5 = SpeechT5Model(config, text_encoder, speech_decoder)
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)
        self.post_init()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSpectrogramOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, decoder_input_values: Optional[torch.FloatTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, speaker_embeddings: Optional[torch.FloatTensor]=None, labels: Optional[torch.FloatTensor]=None, stop_labels: Optional[torch.Tensor]=None) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
        """
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`SpeechT5Tokenizer`]. See [`~PreTrainedTokenizer.encode`] and
            [`~PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        decoder_input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`):
            Float values of input mel spectrogram.

            SpeechT5 uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
            `past_key_values`).
        speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
            Tensor containing the speaker embeddings.
        labels (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*):
            Float values of target mel spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
            computation. Spectrograms can be obtained using [`SpeechT5Processor`]. See [`SpeechT5Processor.__call__`]
            for details.

        Returns:

        Example:

        ```python
        >>> from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
        >>> import torch

        >>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        >>> model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        >>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
        >>> speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file

        >>> set_seed(555)  # make deterministic

        >>> # generate speech
        >>> speech = model.generate(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        >>> speech.shape
        torch.Size([15872])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values = shift_spectrograms_right(labels, self.config.reduction_factor)
            if self.config.use_guided_attention_loss:
                output_attentions = True
        outputs = self.speecht5(input_values=input_ids, attention_mask=attention_mask, decoder_input_values=decoder_input_values, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, use_cache=use_cache, speaker_embeddings=speaker_embeddings, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        outputs_before_postnet, outputs_after_postnet, logits = self.speech_decoder_postnet(outputs[0])
        loss = None
        if labels is not None:
            criterion = SpeechT5SpectrogramLoss(self.config)
            loss = criterion(attention_mask, outputs_before_postnet, outputs_after_postnet, logits, labels, outputs.cross_attentions)
        if not return_dict:
            output = (outputs_after_postnet,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Seq2SeqSpectrogramOutput(loss=loss, spectrogram=outputs_after_postnet, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor]=None, speaker_embeddings: Optional[torch.FloatTensor]=None, threshold: float=0.5, minlenratio: float=0.0, maxlenratio: float=20.0, vocoder: Optional[nn.Module]=None, output_cross_attentions: bool=False, return_output_lengths: bool=False, **kwargs) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a
        speech waveform using a vocoder.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SpeechT5Tokenizer`]. See [`~PreTrainedTokenizer.encode`] and
                [`~PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Attention mask from the tokenizer, required for batched inference to signal to the model where to
                ignore padded tokens from the input_ids.
            speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
                Tensor containing the speaker embeddings.
            threshold (`float`, *optional*, defaults to 0.5):
                The generated sequence ends when the predicted stop token probability exceeds this value.
            minlenratio (`float`, *optional*, defaults to 0.0):
                Used to calculate the minimum required length for the output sequence.
            maxlenratio (`float`, *optional*, defaults to 20.0):
                Used to calculate the maximum allowed length for the output sequence.
            vocoder (`nn.Module`, *optional*):
                The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
                spectrogram.
            output_cross_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of the decoder's cross-attention layers.
            return_output_lengths (`bool`, *optional*, defaults to `False`):
                Whether or not to return the concrete spectrogram/waveform lengths.

        Returns:
            `tuple(torch.FloatTensor)` comprising various elements depending on the inputs:
            - when `return_output_lengths` is False
                - **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
                - **waveform** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(num_frames,)` -- The predicted speech waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
            - when `return_output_lengths` is True
                - **spectrograms** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrograms that
                are padded to the maximum length.
                - **spectrogram_lengths** (*optional*, returned when no `vocoder` is provided) `List[Int]` -- A list of
                all the concrete lengths for each spectrogram.
                - **waveforms** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, num_frames)` -- The predicted speech waveforms that are padded to the maximum length.
                - **waveform_lengths** (*optional*, returned when a `vocoder` is provided) `List[Int]` -- A list of all
                the concrete lengths for each waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
        """
        if speaker_embeddings is not None:
            batch_size = input_ids.size(0)
            if speaker_embeddings.size(0) != batch_size:
                if speaker_embeddings.size(0) == 1:
                    speaker_embeddings = speaker_embeddings.repeat(batch_size, 1)
                else:
                    raise ValueError('The first dimension of speaker_embeddings must be either 1 or the same as batch_size.')
        return _generate_speech(self, input_ids, speaker_embeddings, attention_mask, threshold, minlenratio, maxlenratio, vocoder, output_cross_attentions, return_output_lengths)

    @torch.no_grad()
    def generate_speech(self, input_ids: torch.LongTensor, speaker_embeddings: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, threshold: float=0.5, minlenratio: float=0.0, maxlenratio: float=20.0, vocoder: Optional[nn.Module]=None, output_cross_attentions: bool=False, return_output_lengths: bool=False) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a
        speech waveform using a vocoder.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SpeechT5Tokenizer`]. See [`~PreTrainedTokenizer.encode`] and
                [`~PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
                Tensor containing the speaker embeddings.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            threshold (`float`, *optional*, defaults to 0.5):
                The generated sequence ends when the predicted stop token probability exceeds this value.
            minlenratio (`float`, *optional*, defaults to 0.0):
                Used to calculate the minimum required length for the output sequence.
            maxlenratio (`float`, *optional*, defaults to 20.0):
                Used to calculate the maximum allowed length for the output sequence.
            vocoder (`nn.Module`, *optional*, defaults to `None`):
                The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
                spectrogram.
            output_cross_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of the decoder's cross-attention layers.
            return_output_lengths (`bool`, *optional*, defaults to `False`):
                Whether or not to return the concrete spectrogram/waveform lengths.

        Returns:
            `tuple(torch.FloatTensor)` comprising various elements depending on the inputs:
            - when `return_output_lengths` is False
                - **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
                - **waveform** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(num_frames,)` -- The predicted speech waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
            - when `return_output_lengths` is True
                - **spectrograms** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrograms that
                are padded to the maximum length.
                - **spectrogram_lengths** (*optional*, returned when no `vocoder` is provided) `List[Int]` -- A list of
                all the concrete lengths for each spectrogram.
                - **waveforms** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
                `(batch_size, num_frames)` -- The predicted speech waveforms that are padded to the maximum length.
                - **waveform_lengths** (*optional*, returned when a `vocoder` is provided) `List[Int]` -- A list of all
                the concrete lengths for each waveform.
                - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
                `torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.decoder_attention_heads,
                output_sequence_length, input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
        """
        if speaker_embeddings is not None:
            batch_size = input_ids.size(0)
            if speaker_embeddings.size(0) != batch_size:
                if speaker_embeddings.size(0) == 1:
                    speaker_embeddings = speaker_embeddings.repeat(batch_size, 1)
                else:
                    raise ValueError('The first dimension of speaker_embeddings must be either 1 or the same as batch size.')
        return _generate_speech(self, input_ids, speaker_embeddings, attention_mask, threshold, minlenratio, maxlenratio, vocoder, output_cross_attentions, return_output_lengths)