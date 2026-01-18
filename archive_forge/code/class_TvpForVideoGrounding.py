import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
@add_start_docstrings('\n    Tvp Model with a video grounding head on top computing IoU, distance, and duration loss.\n    ', TVP_START_DOCSTRING)
class TvpForVideoGrounding(TvpPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = TvpModel(config)
        self.video_grounding_head = TvpVideoGroundingHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(TVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvpVideoGroundingOutput, config_class=TvpConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, labels: Tuple[torch.Tensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        """
        labels (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*):
            The labels contains duration, start time, and end time of the video corresponding to the text.
        Returns:

        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoConfig, AutoTokenizer, TvpForVideoGrounding

        >>> model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp")

        >>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

        >>> pixel_values = torch.rand(1, 1, 3, 448, 448)
        >>> text_inputs = tokenizer("This is an example input", return_tensors="pt")
        >>> output = model(text_inputs.input_ids, pixel_values, text_inputs.attention_mask)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.model(input_ids, pixel_values, attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooler_output = outputs[1]
        logits = self.video_grounding_head(pooler_output)
        loss = None
        if labels is not None:
            criterion = TvpLoss(['iou', 'distance', 'duration'])
            criterion.to(self.device)
            loss_dict = criterion(logits, labels)
            loss = loss_dict['iou'] + self.config.distance_loss_weight * loss_dict['distance'] + self.config.duration_loss_weight * loss_dict['duration']
        if not return_dict:
            outputs = (logits,) + outputs[2:]
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
        return TvpVideoGroundingOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)