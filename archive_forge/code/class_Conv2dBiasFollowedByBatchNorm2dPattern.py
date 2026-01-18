import json
import math
import os
import re
from typing import Dict, List, Optional, Set
import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
class Conv2dBiasFollowedByBatchNorm2dPattern(Pattern):
    """
    This pattern identifies if we are enabling bias in Conv2d which is followed by BatchNorm2d.
    Bias doesn't do anything when followed by batchnorm.
    Pattern:
    nn.Module: Conv2d            | nn.Module: BatchNorm2d
        ...
            aten::conv2d AND dtype of third argument is not null
    The third argument is the bias
    Algorithm:
    String match
    """

    def __init__(self, prof: profile, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.name = 'Enabling Bias in Conv2d Followed By BatchNorm Pattern'
        self.description = "Detected bias enabled in Conv2d that is followed by BatchNorm2d. Please set 'bias=False' in Conv2d."
        self.url = 'https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm'

    @property
    def skip(self):
        return self.prof.record_shapes is False or super().skip

    def match(self, event: _ProfilerEvent):
        if event.name != 'aten::conv2d':
            return False
        if len(input_dtypes(event)) < 3 or input_dtypes(event)[2] is None:
            return False
        event = self.go_up_until(event, lambda e: e.name.startswith('nn.Module: Conv2d'))
        if not event:
            return False
        event = self.next_of(event)
        if not event:
            return False
        return event.name.startswith('nn.Module: BatchNorm2d')