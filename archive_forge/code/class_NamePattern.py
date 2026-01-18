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
class NamePattern(Pattern):

    def __init__(self, prof: profile, name: str, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.description = f'Matched Name Event: {name}'
        self.name = name

    def match(self, event: _ProfilerEvent):
        return re.search(self.name, event.name) is not None