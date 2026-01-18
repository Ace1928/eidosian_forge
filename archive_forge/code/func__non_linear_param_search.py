import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
    assert self.histogram.size()[0] == self.bins, 'bins mismatch'
    bin_width = (self.max_val - self.min_val) / self.bins
    total = torch.sum(self.histogram).item()
    cSum = torch.cumsum(self.histogram, dim=0)
    stepsize = 1e-05
    alpha = 0.0
    beta = 1.0
    start_bin = 0
    end_bin = self.bins - 1
    norm_min = float('inf')
    while alpha < beta:
        next_alpha = alpha + stepsize
        next_beta = beta - stepsize
        l = start_bin
        r = end_bin
        while l < end_bin and cSum[l] < next_alpha * total:
            l = l + 1
        while r > start_bin and cSum[r] > next_beta * total:
            r = r - 1
        next_start_bin = start_bin
        next_end_bin = end_bin
        if l - start_bin > end_bin - r:
            next_start_bin = l
            alpha = next_alpha
        else:
            next_end_bin = r
            beta = next_beta
        if next_start_bin == start_bin and next_end_bin == end_bin:
            continue
        norm = self._compute_quantization_error(next_start_bin, next_end_bin)
        if norm > norm_min:
            break
        norm_min = norm
        start_bin = next_start_bin
        end_bin = next_end_bin
    new_min = self.min_val + bin_width * start_bin
    new_max = self.min_val + bin_width * (end_bin + 1)
    return (new_min, new_max)