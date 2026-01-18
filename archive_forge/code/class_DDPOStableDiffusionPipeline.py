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
class DDPOStableDiffusionPipeline:
    """
    Main class for the diffusers pipeline to be finetuned with the DDPO trainer
    """

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        raise NotImplementedError

    def scheduler_step(self, *args, **kwargs) -> DDPOSchedulerOutput:
        raise NotImplementedError

    @property
    def unet(self):
        """
        Returns the 2d U-Net model used for diffusion.
        """
        raise NotImplementedError

    @property
    def vae(self):
        """
        Returns the Variational Autoencoder model used from mapping images to and from the latent space
        """
        raise NotImplementedError

    @property
    def tokenizer(self):
        """
        Returns the tokenizer used for tokenizing text inputs
        """
        raise NotImplementedError

    @property
    def scheduler(self):
        """
        Returns the scheduler associated with the pipeline used for the diffusion process
        """
        raise NotImplementedError

    @property
    def text_encoder(self):
        """
        Returns the text encoder used for encoding text inputs
        """
        raise NotImplementedError

    @property
    def autocast(self):
        """
        Returns the autocast context manager
        """
        raise NotImplementedError

    def set_progress_bar_config(self, *args, **kwargs):
        """
        Sets the progress bar config for the pipeline
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        """
        Saves all of the model weights
        """
        raise NotImplementedError

    def get_trainable_layers(self, *args, **kwargs):
        """
        Returns the trainable parameters of the pipeline
        """
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        """
        Light wrapper around accelerate's register_save_state_pre_hook which is run before saving state
        """
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        """
        Light wrapper around accelerate's register_lad_state_pre_hook which is run before loading state
        """
        raise NotImplementedError