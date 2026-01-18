import logging
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
from xformers.components.attention.utils import (
@dataclass
class NystromSelfAttentionConfig(AttentionConfig):
    """
    num_heads               Number of heads.
    num_landmarks           Number of landmarks to use for softmax approximation. 64 often sufficient for a good
                            approximation according to https://arxiv.org/pdf/2102.03902.pdf.
    causal                  Apply a causal mask, in that the attention cannot be applied to the future.
    use_razavi_pinverse     If true, use iterative method from (Razavi et al. 2014) to approximate the Moore-Penrose
                            inverse, otherwise use standard torch inverse.
    pinverse_original_init  True if using original initialization when calculating Moore-Penrose pseudo inverse using
                            method from (Razavi et al. 2014).
                            False if using exact coefficient computation (leads to faster convergence).
    inv_iterations          Number of iterations for calculating the Moore-Penrose pseudo inverse.
    v_skip_connection       A module that will take V as input and will be added as a skip connection to the
                            softmax approximation. A skip connection is added in the paper to help with training.
    conv_kernel_size        Kernel size for convolution optionally added to help in training.
                            If v_skip_connection is not specified, this will be used to define the default
                            depth wise convolution used as a skip connection.
                            If both conv_kernel_size and v_skip_connection are None, no skip connection will
                            be added.
    landmark_pooling        Which module to use when computing landmarks. Default is AdaptiveAvgPool2d.
    """
    num_heads: int
    num_landmarks: Optional[int]
    landmark_pooling: Optional[nn.Module]
    causal: Optional[bool]
    pinverse_original_init: Optional[bool]
    inv_iterations: Optional[int]
    v_skip_connection: Optional[nn.Module]
    conv_kernel_size: Optional[int]
    use_razavi_pinverse: Optional[bool]