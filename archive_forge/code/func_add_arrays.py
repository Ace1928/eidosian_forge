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
def add_arrays(self, arrays):
    """
        Add `arrays` to the internal storage, Will initialize the storage to the full size at the first arrays passed
        so that if we're bound to get an OOM, it happens at the beginning.
        """
    if arrays is None:
        return
    if self._storage is None:
        self._storage = nested_new_like(arrays, self.total_samples, padding_index=self.padding_index)
        self._offsets = list(range(0, self.total_samples, self.process_length))
    slice_len, self._storage = self._nested_set_tensors(self._storage, arrays)
    for i in range(self.world_size):
        self._offsets[i] += slice_len