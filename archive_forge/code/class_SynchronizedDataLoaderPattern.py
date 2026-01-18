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
class SynchronizedDataLoaderPattern(Pattern):
    """
    This pattern identifies if we are using num_workers=0 in DataLoader.
    example:
    torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    Add num_workers=N to the arguments. N depends on system configuration.

    Pattern:
    dataloader.py(...): __iter__
        dataloader.py(...): _get_iterator
            NOT dataloader.py(...): check_worker_number_rationality

    Algorithm:
    If we don't see check_worker_number_rationality call in the dataloader __iter__,
    It is not an asynchronous dataloader.

    """

    def __init__(self, prof: profile, should_benchmark: bool=False):
        super().__init__(prof, should_benchmark)
        self.name = 'Synchronized DataLoader Pattern'
        self.description = 'Detected DataLoader running with synchronized implementation. Please enable asynchronous dataloading by setting num_workers > 0 when initializing DataLoader.'
        self.url = 'https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation'

    def match(self, event: _ProfilerEvent):

        def is_dataloader_function(name: str, function_name: str):
            return name.startswith(os.path.join('torch', 'utils', 'data', 'dataloader.py')) and name.endswith(function_name)
        try:
            event.name
        except UnicodeDecodeError:
            return False
        if not is_dataloader_function(event.name, '__iter__'):
            return False
        if not event.children:
            return False
        event = event.children[0]
        if not is_dataloader_function(event.name, '_get_iterator'):
            return False
        if not event.children:
            return False
        event = event.children[0]
        return not is_dataloader_function(event.name, 'check_worker_number_rationality')