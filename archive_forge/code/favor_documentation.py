import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.feature_maps import (

        Kernelized attention, as proposed in Performers_
        ("Rethinking attention with performers." K. Choromanski et al. (2020).).

        FAVOR stands for "Fast Attention Via positive Orthogonal Random features"

        Args:
            dropout (float): the probability of an output to be randomly dropped at training time
            dim_features (int): the dimension of the random features space
            iter_before_redraw (int): the number of steps (forward calls) before a redraw of the features
            feature_map_type (FeatureMapType): the type of feature map being used,
            for instance orthogonal random features.

        .. _Performers: https://arxiv.org/pdf/2009.14794v1.pdf
        