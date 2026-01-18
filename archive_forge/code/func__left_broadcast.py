import contextlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from ..core import randn_tensor
from ..import_utils import is_peft_available
from .sd_utils import convert_state_dict_to_diffusers
def _left_broadcast(input_tensor, shape):
    """
    As opposed to the default direction of broadcasting (right to left), this function broadcasts
    from left to right
        Args:
            input_tensor (`torch.FloatTensor`): is the tensor to broadcast
            shape (`Tuple[int]`): is the shape to broadcast to
    """
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError('The number of dimensions of the tensor to broadcast cannot be greater than the length of the shape to broadcast to')
    return input_tensor.reshape(input_tensor.shape + (1,) * (len(shape) - input_ndim)).broadcast_to(shape)