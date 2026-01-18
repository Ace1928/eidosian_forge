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
@add_start_docstrings('The PatchTST for pretrain model.', PATCHTST_START_DOCSTRING)
class PatchTSTForPretraining(PatchTSTPreTrainedModel):

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        config.do_mask_input = True
        self.model = PatchTSTModel(config=config)
        self.head = PatchTSTMaskPretrainHead(config)
        self.post_init()

    def forward(self, past_values: torch.Tensor, past_observed_mask: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, PatchTSTForPretrainingOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPretrainingOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import PatchTSTConfig, PatchTSTForPretraining

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> # Config for random mask pretraining
        >>> config = PatchTSTConfig(
        ...     num_input_channels=7,
        ...     context_length=512,
        ...     patch_length=12,
        ...     stride=12,
        ...     mask_type='random',
        ...     random_mask_ratio=0.4,
        ...     use_cls_token=True,
        ... )
        >>> # Config for forecast mask pretraining
        >>> config = PatchTSTConfig(
        ...     num_input_channels=7,
        ...     context_length=512,
        ...     patch_length=12,
        ...     stride=12,
        ...     mask_type='forecast',
        ...     num_forecast_mask_patches=5,
        ...     use_cls_token=True,
        ... )
        >>> model = PatchTSTForPretraining(config)

        >>> # during training, one provides both past and future values
        >>> outputs = model(past_values=batch["past_values"])

        >>> loss = outputs.loss
        >>> loss.backward()
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_output = self.model(past_values=past_values, past_observed_mask=past_observed_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=True)
        x_hat = self.head(model_output.last_hidden_state)
        loss = nn.MSELoss(reduction='none')
        loss_val = loss(x_hat, model_output.patch_input)
        masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)
        encoder_states = model_output.hidden_states
        if not return_dict:
            outputs = (x_hat,) + model_output[1:-4]
            outputs = (masked_loss,) + outputs if masked_loss is not None else outputs
            return outputs
        return PatchTSTForPretrainingOutput(loss=masked_loss, prediction_output=x_hat, hidden_states=encoder_states, attentions=model_output.attentions)