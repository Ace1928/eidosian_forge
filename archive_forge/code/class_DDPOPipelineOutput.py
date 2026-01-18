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
@dataclass
class DDPOPipelineOutput:
    """
    Output class for the diffusers pipeline to be finetuned with the DDPO trainer

    Args:
        images (`torch.Tensor`):
            The generated images.
        latents (`List[torch.Tensor]`):
            The latents used to generate the images.
        log_probs (`List[torch.Tensor]`):
            The log probabilities of the latents.

    """
    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor