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
@add_start_docstrings('The PatchTST for regression model.', PATCHTST_START_DOCSTRING)
class PatchTSTForRegression(PatchTSTPreTrainedModel):

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        if config.do_mask_input:
            logger.warning('Setting `do_mask_input` parameter to False.')
            config.do_mask_input = False
        self.model = PatchTSTModel(config)
        if config.loss == 'mse':
            self.distribution_output = None
        elif config.distribution_output == 'student_t':
            self.distribution_output = StudentTOutput(dim=config.num_targets)
        elif config.distribution_output == 'normal':
            self.distribution_output = NormalOutput(dim=config.num_targets)
        elif config.distribution_output == 'negative_binomial':
            self.distribution_output = NegativeBinomialOutput(dim=config.num_targets)
        else:
            raise ValueError(f'Unknown distribution output {config.distribution_output}')
        self.head = PatchTSTRegressionHead(config, self.distribution_output)
        self.post_init()

    def forward(self, past_values: torch.Tensor, target_values: torch.Tensor=None, past_observed_mask: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, PatchTSTForRegressionOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            target_values (`torch.Tensor` of shape `(bs, num_input_channels)`):
                Target values associates with the `past_values`
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
            `PatchTSTForRegressionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from transformers import PatchTSTConfig, PatchTSTForRegression

        >>> # Regression task with 6 input channels and regress 2 targets
        >>> model = PatchTSTForRegression.from_pretrained("namctin/patchtst_etth1_regression")

        >>> # during inference, one only provides past values, the model outputs future values
        >>> past_values = torch.randn(20, 512, 6)
        >>> outputs = model(past_values=past_values)
        >>> regression_outputs = outputs.regression_outputs
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_output = self.model(past_values=past_values, past_observed_mask=past_observed_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=True)
        y_hat = self.head(model_output.last_hidden_state)
        loss = None
        if target_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(y_hat)
                y_hat = tuple([item.view(-1, self.config.num_targets) for item in y_hat])
                loss = nll(distribution, target_values)
                loss = weighted_average(loss)
            else:
                loss = nn.MSELoss(reduction='mean')
                loss = loss(y_hat, target_values)
        if not return_dict:
            outputs = (y_hat,) + model_output[1:-3]
            outputs = (loss,) + outputs if loss is not None else outputs
            return outputs
        return PatchTSTForRegressionOutput(loss=loss, regression_outputs=y_hat, hidden_states=model_output.hidden_states, attentions=model_output.attentions)

    def generate(self, past_values: torch.Tensor, past_observed_mask: Optional[torch.Tensor]=None) -> SamplePatchTSTOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
            samples, num_targets)`.
        """
        num_parallel_samples = self.config.num_parallel_samples
        outputs = self(past_values=past_values, target_values=None, past_observed_mask=past_observed_mask, output_hidden_states=False)
        distribution = self.distribution_output.distribution(outputs.regression_outputs)
        samples = [distribution.sample() for _ in range(num_parallel_samples)]
        samples = torch.stack(samples, dim=1).view(-1, num_parallel_samples, self.config.num_targets)
        return SamplePatchTSTOutput(sequences=samples)