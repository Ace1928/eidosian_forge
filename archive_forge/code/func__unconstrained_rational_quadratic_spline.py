import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
def _unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, reverse=False, tail_bound=5.0, min_bin_width=0.001, min_bin_height=0.001, min_derivative=0.001):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Outside of the
    `tail_bound`, the transform behaves as an identity function.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`, *optional*, defaults to `False`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`, *optional* defaults to 5):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function with the `tail_bound` limits
            applied.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs` with the `tail_bound`
            limits applied.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask
    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    log_abs_det[outside_interval_mask] = 0.0
    outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _rational_quadratic_spline(inputs=inputs[inside_interval_mask], unnormalized_widths=unnormalized_widths[inside_interval_mask, :], unnormalized_heights=unnormalized_heights[inside_interval_mask, :], unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :], reverse=reverse, tail_bound=tail_bound, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative)
    return (outputs, log_abs_det)