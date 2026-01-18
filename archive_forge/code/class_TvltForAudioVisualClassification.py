import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
@add_start_docstrings('\n    Tvlt Model transformer with a classifier head on top (an MLP on top of the final hidden state of the [CLS] token)\n    for audiovisual classification tasks, e.g. CMU-MOSEI Sentiment Analysis and Audio to Video Retrieval.\n    ', TVLT_START_DOCSTRING)
class TvltForAudioVisualClassification(TvltPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.tvlt = TvltModel(config)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size * 2), nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps), nn.GELU(), nn.Linear(config.hidden_size * 2, config.num_labels))
        self.config = config
        self.post_init()

    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, audio_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor]=None, audio_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the audiovisual loss. Indices should be in `[0, ..., num_classes-1]` where num_classes
            refers to the number of classes in audiovisual tasks.

        Return:

        Examples:
        ```python
        >>> from transformers import TvltProcessor, TvltForAudioVisualClassification
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))
        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltForAudioVisualClassification.from_pretrained("ZinengTang/tvlt-base")
        >>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.tvlt(pixel_values, audio_values, pixel_mask=pixel_mask, audio_mask=audio_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0][:, 0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.config.loss_type == 'regression':
                loss_fct = MSELoss()
                loss = loss_fct(logits, labels)
            elif self.config.loss_type == 'classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[4:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)