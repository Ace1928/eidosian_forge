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
@dataclass
class OrthoformerAttentionConfig(AttentionConfig):
    """
    num_landmarks           Number of landmarks to use for softmax approximation.
    subsample_fraction      Percentage of q_samples matrix to sample per iteration
    landmark_selection      Landmark selection strategy
    """
    num_landmarks: Optional[int]
    subsample_fraction: Optional[float]
    landmark_selection: Optional[LandmarkSelection]