import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    """
    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config)
        self.loss = config.loss
        self.distribution_output = config.distribution_output
        self.use_return_dict = config.use_return_dict
        self.num_parallel_samples = config.num_parallel_samples
        if config.loss == 'mse':
            self.distribution_output = None
        else:
            distribution_output_map = {'student_t': StudentTOutput, 'normal': NormalOutput, 'negative_binomial': NegativeBinomialOutput}
            output_class = distribution_output_map.get(config.distribution_output)
            if output_class is not None:
                self.distribution_output = output_class(dim=config.num_targets)
            else:
                raise ValueError(f'Unknown distribution output {config.distribution_output}')
        if config.scaling in ['std', 'mean', True]:
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None
        self.head = PatchTSMixerLinearHead(config=config, distribution_output=self.distribution_output)
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, target_values: torch.Tensor=None, output_hidden_states: Optional[bool]=False, return_loss: bool=True, return_dict: Optional[bool]=None) -> PatchTSMixerForRegressionOutput:
        """
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        if self.loss == 'mse':
            loss = nn.MSELoss(reduction='mean')
        elif self.loss == 'nll':
            loss = nll
        else:
            raise ValueError('Invalid loss function: Allowed values: mse and nll')
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        model_output = self.model(past_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)
        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(model_output.last_hidden_state, loc=model_output.loc, scale=model_output.scale)
        y_hat = self.head(model_output.last_hidden_state)
        if target_values is not None and return_loss is True:
            if self.distribution_output:
                if self.distribution_output == 'negative_binomial' and torch.any(target_values < 0):
                    raise Exception('target_values cannot be negative for negative_binomial distribution.')
                distribution = self.distribution_output.distribution(y_hat)
                y_hat = tuple([item.view(-1, self.config.num_targets) for item in y_hat])
                loss_val = loss(distribution, target_values)
                loss_val = weighted_average(loss_val)
            else:
                loss_val = loss(y_hat, target_values)
        else:
            loss_val = None
        if not return_dict:
            return tuple((v for v in [loss_val, y_hat, model_output.last_hidden_state, model_output.hidden_states]))
        return PatchTSMixerForRegressionOutput(loss=loss_val, regression_outputs=y_hat, last_hidden_state=model_output.last_hidden_state, hidden_states=model_output.hidden_states)

    def generate(self, past_values: torch.Tensor) -> SamplePatchTSMixerRegressionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the target values.

        Return:
            [`SamplePatchTSMixerRegressionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, num_targets)`.
        """
        num_parallel_samples = self.num_parallel_samples
        outputs = self(past_values=past_values, target_values=None, output_hidden_states=False)
        distribution = self.distribution_output.distribution(outputs.regression_outputs)
        samples = [distribution.sample() for _ in range(num_parallel_samples)]
        samples = torch.stack(samples, dim=1).view(-1, num_parallel_samples, self.config.num_targets)
        return SamplePatchTSMixerRegressionOutput(sequences=samples)