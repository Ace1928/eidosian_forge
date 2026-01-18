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
class ForLoopIndexingPattern(Pattern):
    """
    This pattern identifies if we use a for loop to index a tensor that
    can be vectorized.
    example:
    tensor = torch.empty((100, 100))
    for i in range(100):
        tensor[i] = i

    Pattern:
    aten::select | ... | aten::select | ... (Repeat)

    Algorithm:
    We start at node aten::select, and we check if we can find this alternating patterns.
    We also keep a dictionary to avoid duplicate match in the for loop.
    """

    def __init__(self, prof: profile, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.name = 'For Loop Indexing Pattern'
        self.description = 'For loop indexing detected. Vectorization recommended.'
        self.visited: Set[int] = set()

    def eventTreeTraversal(self):
        """
        We need to use BFS traversal order to avoid duplicate match.
        """
        yield from traverse_bfs(self.event_tree)

    def match(self, event: _ProfilerEvent):
        if event.name != 'aten::select':
            return False
        if event.id in self.visited:
            return False
        repeat_count = 1
        _, next = self.siblings_of(event)
        if len(next) <= 1:
            return False

        def same_ops(list1, list2):
            if len(list1) != len(list2):
                return False
            for op1, op2 in zip(list1, list2):
                if op1.name != op2.name:
                    return False
            return True
        next_select_idx = index_of_first_match(next, lambda e: e.name == 'aten::select')
        if next_select_idx is None:
            return False
        indexing_ops = [event] + next[:next_select_idx]
        next = next[len(indexing_ops) - 1:]
        for i in range(0, len(next), len(indexing_ops)):
            if same_ops(indexing_ops, next[i:i + len(indexing_ops)]):
                repeat_count += 1
                self.visited.add(next[i].id)
            else:
                break
        return repeat_count >= 10