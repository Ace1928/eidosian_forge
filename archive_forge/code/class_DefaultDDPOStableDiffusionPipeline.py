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
class DefaultDDPOStableDiffusionPipeline(DDPOStableDiffusionPipeline):

    def __init__(self, pretrained_model_name: str, *, pretrained_model_revision: str='main', use_lora: bool=True):
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name, revision=pretrained_model_revision)
        self.use_lora = use_lora
        self.pretrained_model = pretrained_model_name
        self.pretrained_revision = pretrained_model_revision
        try:
            self.sd_pipeline.load_lora_weights(pretrained_model_name, weight_name='pytorch_lora_weights.safetensors', revision=pretrained_model_revision)
            self.use_lora = True
        except OSError:
            if use_lora:
                warnings.warn('If you are aware that the pretrained model has no lora weights to it, ignore this message. Otherwise please check the if `pytorch_lora_weights.safetensors` exists in the model folder.')
        self.sd_pipeline.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config)
        self.sd_pipeline.safety_checker = None
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(not self.use_lora)

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        return pipeline_step(self.sd_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs) -> DDPOSchedulerOutput:
        return scheduler_step(self.sd_pipeline.scheduler, *args, **kwargs)

    @property
    def unet(self):
        return self.sd_pipeline.unet

    @property
    def vae(self):
        return self.sd_pipeline.vae

    @property
    def tokenizer(self):
        return self.sd_pipeline.tokenizer

    @property
    def scheduler(self):
        return self.sd_pipeline.scheduler

    @property
    def text_encoder(self):
        return self.sd_pipeline.text_encoder

    @property
    def autocast(self):
        return contextlib.nullcontext if self.use_lora else None

    def save_pretrained(self, output_dir):
        if self.use_lora:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.sd_pipeline.unet))
            self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        self.sd_pipeline.save_pretrained(output_dir)

    def set_progress_bar_config(self, *args, **kwargs):
        self.sd_pipeline.set_progress_bar_config(*args, **kwargs)

    def get_trainable_layers(self):
        if self.use_lora:
            lora_config = LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian', target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'])
            self.sd_pipeline.unet.add_adapter(lora_config)
            for param in self.sd_pipeline.unet.parameters():
                if param.requires_grad:
                    param.data = param.to(torch.float32)
            return self.sd_pipeline.unet
        else:
            return self.sd_pipeline.unet

    def save_checkpoint(self, models, weights, output_dir):
        if len(models) != 1:
            raise ValueError('Given how the trainable params were set, this should be of length 1')
        if self.use_lora and hasattr(models[0], 'peft_config') and (getattr(models[0], 'peft_config', None) is not None):
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            self.sd_pipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)
        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, 'unet'))
        else:
            raise ValueError(f'Unknown model type {type(models[0])}')

    def load_checkpoint(self, models, input_dir):
        if len(models) != 1:
            raise ValueError('Given how the trainable params were set, this should be of length 1')
        if self.use_lora:
            lora_state_dict, network_alphas = self.sd_pipeline.lora_state_dict(input_dir, weight_name='pytorch_lora_weights.safetensors')
            self.sd_pipeline.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=models[0])
        elif not self.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder='unet')
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f'Unknown model type {type(models[0])}')