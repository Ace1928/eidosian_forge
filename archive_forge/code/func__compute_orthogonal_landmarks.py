import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as Fn
from xformers.components.attention import (
from xformers.components.attention.core import (
def _compute_orthogonal_landmarks(self, q: torch.Tensor) -> torch.Tensor:
    """
        Construct set of landmarks by recursively selecting new landmarks
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """
    if self.subsample_fraction < 1.0:
        num_samples = max(int(self.subsample_fraction * q.size(-2)), self.num_landmarks)
        q_samples = q[:, torch.randint(q.size(-2), (num_samples,), device=q.device), :]
    else:
        q_samples = q
    q_samples_normalized = Fn.normalize(q_samples, p=2, dim=-1)
    B, N, D = q_samples_normalized.shape
    selected_mask = torch.zeros((B, N, 1), device=q_samples_normalized.device)
    landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=q_samples_normalized.device)
    random_idx = torch.randint(q_samples_normalized.size(-2), (B, 1, 1), device=q_samples_normalized.device)
    selected_mask.scatter_(-2, random_idx, landmark_mask)
    selected_landmarks = torch.empty((B, self.num_landmarks, D), device=q_samples_normalized.device, dtype=q_samples_normalized.dtype)
    selected_landmarks[:, 0, :] = q_samples_normalized[torch.arange(q_samples_normalized.size(0)), random_idx.view(-1), :].view(B, D)
    cos_sims = torch.empty((B, N, self.num_landmarks), device=q_samples_normalized.device, dtype=q_samples_normalized.dtype)
    for M in range(1, self.num_landmarks):
        with profiler.record_function('find new landmark'):
            cos_sims[:, :, M - 1] = torch.einsum('b n d, b d -> b n', q_samples_normalized, selected_landmarks[:, M - 1, :]).abs()
            cos_sim_set = cos_sims[:, :, :M]
            cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
            selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
            selected_landmarks[:, M, :] = q_samples_normalized[torch.arange(q_samples_normalized.size(0)), selected_landmark_idx, :].view(B, D)
            selected_mask.scatter_(-2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask)
    landmarks = torch.masked_select(q_samples, selected_mask.bool()).reshape(B, -1, D)
    return landmarks