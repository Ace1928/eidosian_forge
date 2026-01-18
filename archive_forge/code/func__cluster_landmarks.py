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
def _cluster_landmarks(self, q: torch.Tensor, spherical: bool=False, num_iters: int=6) -> torch.Tensor:
    """
        Construct set of landmarks by recursively selecting new landmarks
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        """
    num_landmarks = min(self.num_landmarks, q.shape[1])
    if self.subsample_fraction < 1.0:
        num_samples = max(int(self.subsample_fraction * q.size(-2)), num_landmarks)
        q_samples = q[:, torch.randint(q.size(-2), (num_samples,)), :]
    else:
        q_samples = q
    if spherical:
        q_samples_normalized = Fn.normalize(q_samples, p=2, dim=-1)
        landmarks = self._kmeans_spherical(q_samples_normalized, num_landmarks, num_iters)
    else:
        landmarks = self._kmeans(q_samples, num_landmarks, num_iters)
    return landmarks