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
class PatchTSMixerForPrediction(PatchTSMixerPreTrainedModel):
    """
    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.loss = config.loss
        self.use_return_dict = config.use_return_dict
        self.prediction_channel_indices = config.prediction_channel_indices
        self.num_parallel_samples = config.num_parallel_samples
        if config.loss == 'mse':
            self.distribution_output = None
        else:
            dim = config.prediction_length
            distribution_output_map = {'student_t': StudentTOutput, 'normal': NormalOutput, 'negative_binomial': NegativeBinomialOutput}
            output_class = distribution_output_map.get(config.distribution_output, None)
            if output_class is not None:
                self.distribution_output = output_class(dim=dim)
            else:
                raise ValueError(f'Unknown distribution output {config.distribution_output}')
        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerForPredictionHead(config=config, distribution_output=self.distribution_output)
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=False, return_loss: bool=True, return_dict: Optional[bool]=None) -> PatchTSMixerForPredictionOutput:
        """
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        future_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,:
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.
        return_loss (`bool`,  *optional*):
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
        model_output = self.model(past_values, observed_mask=observed_mask, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)
        y_hat = self.head(model_output.last_hidden_state)
        loss_val = None
        if self.prediction_channel_indices is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(y_hat, loc=model_output.loc[..., self.prediction_channel_indices], scale=model_output.scale[..., self.prediction_channel_indices])
                if future_values is not None and return_loss is True:
                    loss_val = loss(distribution, future_values[..., self.prediction_channel_indices])
                    loss_val = weighted_average(loss_val)
            else:
                y_hat = y_hat * model_output.scale[..., self.prediction_channel_indices] + model_output.loc[..., self.prediction_channel_indices]
                if future_values is not None and return_loss is True:
                    loss_val = loss(y_hat, future_values[..., self.prediction_channel_indices])
        elif self.distribution_output:
            distribution = self.distribution_output.distribution(y_hat, loc=model_output.loc, scale=model_output.scale)
            if future_values is not None and return_loss is True:
                loss_val = loss(distribution, future_values)
                loss_val = weighted_average(loss_val)
        else:
            y_hat = y_hat * model_output.scale + model_output.loc
            if future_values is not None and return_loss is True:
                loss_val = loss(y_hat, future_values)
        if self.prediction_channel_indices is not None:
            loc = model_output.loc[..., self.prediction_channel_indices]
            scale = model_output.scale[..., self.prediction_channel_indices]
        else:
            loc = model_output.loc
            scale = model_output.scale
        if not return_dict:
            return tuple((v for v in [loss_val, y_hat, model_output.last_hidden_state, model_output.hidden_states, loc, scale]))
        return PatchTSMixerForPredictionOutput(loss=loss_val, prediction_outputs=y_hat, last_hidden_state=model_output.last_hidden_state, hidden_states=model_output.hidden_states, loc=loc, scale=scale)

    def generate(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor]=None) -> SamplePatchTSMixerPredictionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSMixerPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_input_channels)`.
        """
        num_parallel_samples = self.num_parallel_samples
        outputs = self(past_values=past_values, future_values=None, observed_mask=observed_mask, output_hidden_states=False)
        distribution = self.distribution_output.distribution(outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale)
        samples = [distribution.sample() for _ in range(num_parallel_samples)]
        samples = torch.stack(samples, dim=1)
        return SamplePatchTSMixerPredictionOutput(sequences=samples)