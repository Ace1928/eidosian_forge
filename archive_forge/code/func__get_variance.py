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
def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(prev_timestep.cpu() >= 0, self.alphas_cumprod.gather(0, prev_timestep.cpu()), self.final_alpha_cumprod).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance