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
class GradNotSetToNonePattern(Pattern):
    """
    This pattern identifies if we are not setting grad to None in zero_grad.
    example:
    optimizer.zero_grad()
    By setting set_to_none=True, we can gain speedup

    Pattern:
    XXXXX: _zero_grad
        NOT aten::zeros
            aten::zero_

    aten::zero_ is called on each parameter in the model.
    We also want to make sure it is not called by aten::zeros.

    Algorithm:
    String match
    """

    def __init__(self, prof: profile, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.name = 'Gradient Set To Zero Instead of None Pattern'
        self.description = "Detected gradient set to zero instead of None. Please add 'set_to_none=True' when calling zero_grad()."
        self.url = 'https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-gradient-calculation-for-validation-or-inference'

    def match(self, event: _ProfilerEvent):
        if not event.name.endswith(': zero_grad'):
            return False
        if not event.children:
            return False
        for sub_event in traverse_dfs(event.children):
            if sub_event.name == 'aten::zero_' and sub_event.parent.name != 'aten::zeros':
                return True
        return False