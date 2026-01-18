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
class ExtraCUDACopyPattern(Pattern):
    """
    This pattern identifies if we creates a constant tensor on CPU and immediately moves it to GPU.
    example: torch.zeros((100, 100)).to("cuda")

    Pattern:
    build-in method                 |build-in method
        ...                         |    aten::to
            aten::fill_/aten::zero_ |        aten::_to_copy

    Algorithm:
    We start at node aten::to, go parent events' previous events,
    and check if we have a aten::fill_/aten::zero_ as we keep going down the tree.
    We always select the last child in the children list when we go down the tree.
    If at any step we failed, it is not a match.
    """

    def __init__(self, prof: profile, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.name = 'Extra CUDA Copy Pattern'
        self.description = 'Filled a CPU tensor and immediately moved it to GPU. Please initialize it on GPU.'
        self.url = 'https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#create-tensors-directly-on-the-target-device'
        self.init_ops = {'aten::fill_', 'aten::zero_', 'aten::normal_', 'aten::uniform_'}

    @property
    def skip(self):
        return not self.prof.with_stack or not self.prof.record_shapes

    def match(self, event):
        if event.name != 'aten::to':
            return False
        to_event = event
        if not event.children:
            return False
        event = event.children[-1]
        if event.name != 'aten::_to_copy':
            return False
        if not event.children:
            return False
        event = event.children[-1]
        if event.name != 'aten::copy_':
            return False
        dtypes = input_dtypes(event)
        if len(dtypes) < 2:
            return False
        if dtypes[0] is None or dtypes[0] != dtypes[1]:
            return False
        event = to_event
        event = event.parent
        if event is None:
            return False
        event = self.prev_of(event)
        if event is None:
            return False
        while event.children:
            event = event.children[-1]
            if event.name in self.init_ops:
                return True
        return event.name in self.init_ops

    def benchmark(self, events: List[_ProfilerEvent]):
        shapes_factor_map = {input_shapes(event): 0.0 for event in events}
        for shape in shapes_factor_map:
            size = shape[0]
            to_timer = benchmark.Timer(stmt='torch.ones(size).to("cuda")', globals={'size': size})
            de_timer = benchmark.Timer(stmt='torch.ones(size, device="cuda")', globals={'size': size})
            to_time = to_timer.timeit(10).mean
            de_time = de_timer.timeit(10).mean
            shapes_factor_map[shape] = de_time / to_time
        return shapes_factor_map