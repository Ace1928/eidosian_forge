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
@dataclass
class AcceleratorConfig:
    """
    A subset of arguments relating to the underlying [`accelerate.Accelerator`]
    implementation utilized in the `Trainer` that can be customized.
    Mostly relating to data.

    Parameters:
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
            round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
            in your script multiplied by the number of processes.
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        use_seedable_sampler (`bool`, *optional*, defaults to `True`):
            Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`]). Ensures
            training results are fully reproducable using a different sampling technique. While seed-to-seed results
            may differ, on average the differences are neglible when using multiple different seeds to compare. Should
            also be ran with [`~utils.set_seed`] for the best results.

    """
    split_batches: bool = field(default=False, metadata={'help': 'Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set in your script multiplied by the number of processes.'})
    dispatch_batches: bool = field(default=None, metadata={'help': 'If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose underlying dataset is an `IterableDataslet`, `False` otherwise.'})
    even_batches: bool = field(default=True, metadata={'help': 'If set to `True`, in cases where the total batch size across all processes does not exactly divide the dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among all workers.'})
    use_seedable_sampler: bool = field(default=True, metadata={'help': 'Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`]).Ensures training results are fully reproducable using a different sampling technique. While seed-to-seed results may differ, on average the differences are neglible when usingmultiple different seeds to compare. Should also be ran with [`~utils.set_seed`] for the best results.'})

    @classmethod
    def from_json_file(cls, json_file):
        open_file = io.open if os.path.exists(json_file) else open
        with open_file(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        extra_keys = sorted((key for key in config_dict.keys() if key not in cls.__dataclass_fields__.keys()))
        if len(extra_keys) > 0:
            raise ValueError(f'The config file at {json_file} had unknown keys ({extra_keys}), please try upgrading your `transformers` version or fix (and potentially remove these keys) from your config file.')
        return cls(**config_dict)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)