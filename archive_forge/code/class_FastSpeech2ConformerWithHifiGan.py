import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
@add_start_docstrings('The FastSpeech2ConformerModel with a FastSpeech2ConformerHifiGan vocoder head that performs text-to-speech (waveform).', FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING)
class FastSpeech2ConformerWithHifiGan(PreTrainedModel):
    config_class = FastSpeech2ConformerWithHifiGanConfig

    def __init__(self, config: FastSpeech2ConformerWithHifiGanConfig):
        super().__init__(config)
        self.model = FastSpeech2ConformerModel(config.model_config)
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)
        self.config = config

    @replace_return_docstrings(output_type=FastSpeech2ConformerWithHifiGanOutput, config_class=FastSpeech2ConformerWithHifiGanConfig)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor]=None, spectrogram_labels: Optional[torch.FloatTensor]=None, duration_labels: Optional[torch.LongTensor]=None, pitch_labels: Optional[torch.FloatTensor]=None, energy_labels: Optional[torch.FloatTensor]=None, speaker_ids: Optional[torch.LongTensor]=None, lang_ids: Optional[torch.LongTensor]=None, speaker_embedding: Optional[torch.FloatTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Union[Tuple, FastSpeech2ConformerModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`: 0 for tokens that are **masked**, 1 for tokens that are **not masked**.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*, defaults to `None`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*, defaults to `None`):
                Batch of padded durations.
            pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged pitch.
            energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged energy.
            speaker_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Speaker ids used to condition features of speech output by the model.
            lang_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Language ids used to condition features of speech output by the model.
            speaker_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`, *optional*, defaults to `None`):
                Embedding containing conditioning signals for the features of the speech.
            return_dict (`bool`, *optional*, defaults to `None`):
                Whether or not to return a [`FastSpeech2ConformerModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*, defaults to `None`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*, defaults to `None`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     FastSpeech2ConformerTokenizer,
        ...     FastSpeech2ConformerWithHifiGan,
        ... )

        >>> tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        >>> inputs = tokenizer("some text to convert to speech", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]

        >>> model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")
        >>> output_dict = model(input_ids, return_dict=True)
        >>> waveform = output_dict["waveform"]
        >>> print(waveform.shape)
        torch.Size([1, 49664])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.model_config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.model_config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.model_config.output_hidden_states
        model_outputs = self.model(input_ids, attention_mask, spectrogram_labels=spectrogram_labels, duration_labels=duration_labels, pitch_labels=pitch_labels, energy_labels=energy_labels, speaker_ids=speaker_ids, lang_ids=lang_ids, speaker_embedding=speaker_embedding, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        if not return_dict:
            has_missing_labels = spectrogram_labels is None or duration_labels is None or pitch_labels is None or (energy_labels is None)
            if has_missing_labels:
                spectrogram = model_outputs[0]
            else:
                spectrogram = model_outputs[1]
        else:
            spectrogram = model_outputs['spectrogram']
        waveform = self.vocoder(spectrogram)
        if not return_dict:
            return model_outputs + (waveform,)
        return FastSpeech2ConformerWithHifiGanOutput(waveform=waveform, **model_outputs)