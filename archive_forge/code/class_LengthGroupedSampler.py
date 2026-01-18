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
class LengthGroupedSampler(Sampler):
    """
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(self, batch_size: int, dataset: Optional[Dataset]=None, lengths: Optional[List[int]]=None, model_input_name: Optional[str]=None, generator=None):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')
        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding)) or model_input_name not in dataset[0]:
                raise ValueError(f"Can only automatically infer lengths for datasets whose items are dictionaries with an '{model_input_name}' key.")
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info('If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...')
            lengths = lengths.tolist()
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)