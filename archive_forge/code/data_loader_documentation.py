import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (

        Wrapper for the xpl.MpDeviceLoader class that knows the total batch size.

        XLA preloading threads will all call DataLoaderShard's __iter__(). Remove rng_types from DataLoaderShard to
        prevent it from using the XLA device in the preloading threads, and synchronize the RNG once from the main
        thread only.

        **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
        