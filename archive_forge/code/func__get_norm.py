import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _get_norm(self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    """
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
    norm = (delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin) / 3
    return density * norm