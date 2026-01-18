from typing import Callable, Optional, Union
import torch
from .base_structured_sparsifier import BaseStructuredSparsifier
Compute distance across all entries in tensor `t` along all dimension
        except for the one identified by dim.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
        Returns:
            distance (torch.Tensor): distance computed across filtters
        