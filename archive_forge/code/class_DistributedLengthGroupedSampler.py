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
class DistributedLengthGroupedSampler(DistributedSampler):
    """
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """

    def __init__(self, batch_size: int, dataset: Optional[Dataset]=None, num_replicas: Optional[int]=None, rank: Optional[int]=None, seed: int=0, drop_last: bool=False, lengths: Optional[List[int]]=None, model_input_name: Optional[str]=None):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding)) or model_input_name not in dataset[0]:
                raise ValueError(f"Can only automatically infer lengths for datasets whose items are dictionaries with an '{model_input_name}' key.")
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info('If lengths is a torch.Tensor, DistributedLengthGroupedSampler will be slow. Converting lengths to List[int]...')
            lengths = lengths.tolist()
        self.lengths = lengths
        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)
        if not self.drop_last:
            indices += indices[:self.total_size - len(indices)]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)