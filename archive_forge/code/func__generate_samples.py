import os
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from ..models import DDPOStableDiffusionPipeline
from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker
def _generate_samples(self, iterations, batch_size):
    """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (List[Dict[str, torch.Tensor]]), prompt_image_pairs (List[List[Any]])
        """
    samples = []
    prompt_image_pairs = []
    self.sd_pipeline.unet.eval()
    sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)
    for _ in range(iterations):
        prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])
        prompt_ids = self.sd_pipeline.tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.sd_pipeline.tokenizer.model_max_length).input_ids.to(self.accelerator.device)
        prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]
        with self.autocast():
            sd_output = self.sd_pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=sample_neg_prompt_embeds, num_inference_steps=self.config.sample_num_steps, guidance_scale=self.config.sample_guidance_scale, eta=self.config.sample_eta, output_type='pt')
            images = sd_output.images
            latents = sd_output.latents
            log_probs = sd_output.log_probs
        latents = torch.stack(latents, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)
        samples.append({'prompt_ids': prompt_ids, 'prompt_embeds': prompt_embeds, 'timesteps': timesteps, 'latents': latents[:, :-1], 'next_latents': latents[:, 1:], 'log_probs': log_probs, 'negative_prompt_embeds': sample_neg_prompt_embeds})
        prompt_image_pairs.append([images, prompts, prompt_metadata])
    return (samples, prompt_image_pairs)