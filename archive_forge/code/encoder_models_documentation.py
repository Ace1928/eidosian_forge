from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer

        A simple conversion of the CLIPEncoderLayer to its `BetterTransformer` implementation.

        **The implementation is valid only for the vision model, that does not use `causal_attention_mask`.**

        Args:
            layer (`torch.nn.Module`):
                The original `CLIPEncoderLayer` where the weights needs to be retrieved.
        