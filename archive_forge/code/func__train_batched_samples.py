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
def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
    """
        Train on a batch of samples. Main training segment

        Args:
            inner_epoch (int): The current inner epoch
            epoch (int): The current epoch
            global_step (int): The current global step
            batched_samples (List[Dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
    info = defaultdict(list)
    for _i, sample in enumerate(batched_samples):
        if self.config.train_cfg:
            embeds = torch.cat([sample['negative_prompt_embeds'], sample['prompt_embeds']])
        else:
            embeds = sample['prompt_embeds']
        for j in range(self.num_train_timesteps):
            with self.accelerator.accumulate(self.sd_pipeline.unet):
                loss, approx_kl, clipfrac = self.calculate_loss(sample['latents'][:, j], sample['timesteps'][:, j], sample['next_latents'][:, j], sample['log_probs'][:, j], sample['advantages'], embeds)
                info['approx_kl'].append(approx_kl)
                info['clipfrac'].append(clipfrac)
                info['loss'].append(loss)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.trainable_layers.parameters() if not isinstance(self.trainable_layers, list) else self.trainable_layers, self.config.train_max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.accelerator.sync_gradients:
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = self.accelerator.reduce(info, reduction='mean')
                info.update({'epoch': epoch, 'inner_epoch': inner_epoch})
                self.accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)
    return global_step