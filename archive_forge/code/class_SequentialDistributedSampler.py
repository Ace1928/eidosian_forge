import copy
import datetime
import io
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import is_sagemaker_mp_enabled, is_torch_tpu_available, is_training_run_on_sagemaker, logging
class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batch_size=None):
        warnings.warn('SequentialDistributedSampler is deprecated and will be removed in v5 of Transformers.', FutureWarning)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        num_samples = len(self.dataset)
        if batch_size is not None:
            self.num_samples = int(math.ceil(num_samples / (batch_size * num_replicas))) * batch_size
        else:
            self.num_samples = int(math.ceil(num_samples / num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size, f'Indices length {len(indices)} and total size {self.total_size} mismatched'
        indices = indices[self.rank * self.num_samples:(self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples, f'Indices length {len(indices)} and sample number {self.num_samples} mismatched'
        return iter(indices)

    def __len__(self):
        return self.num_samples