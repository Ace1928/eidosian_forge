import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
@add_start_docstrings('The PatchTST for classification model.', PATCHTST_START_DOCSTRING)
class PatchTSTForClassification(PatchTSTPreTrainedModel):

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        if config.do_mask_input:
            logger.warning('Setting `do_mask_input` parameter to False.')
            config.do_mask_input = False
        self.model = PatchTSTModel(config)
        self.head = PatchTSTClassificationHead(config)
        self.post_init()

    def forward(self, past_values: torch.Tensor, target_values: torch.Tensor=None, past_observed_mask: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, PatchTSTForClassificationOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            target_values (`torch.Tensor`, *optional*):
                Labels associates with the `past_values`
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForClassificationOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from transformers import PatchTSTConfig, PatchTSTForClassification

        >>> # classification task with two input channel2 and 3 classes
        >>> config = PatchTSTConfig(
        ...     num_input_channels=2,
        ...     num_targets=3,
        ...     context_length=512,
        ...     patch_length=12,
        ...     stride=12,
        ...     use_cls_token=True,
        ... )
        >>> model = PatchTSTForClassification(config=config)

        >>> # during inference, one only provides past values
        >>> past_values = torch.randn(20, 512, 2)
        >>> outputs = model(past_values=past_values)
        >>> labels = outputs.prediction_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_output = self.model(past_values=past_values, past_observed_mask=past_observed_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=True)
        y_hat = self.head(model_output.last_hidden_state)
        loss_val = None
        if target_values is not None:
            loss = nn.CrossEntropyLoss()
            loss_val = loss(y_hat, target_values)
        if not return_dict:
            outputs = (y_hat,) + model_output[1:-3]
            outputs = (loss_val,) + outputs if loss_val is not None else outputs
            return outputs
        return PatchTSTForClassificationOutput(loss=loss_val, prediction_logits=y_hat, hidden_states=model_output.hidden_states, attentions=model_output.attentions)