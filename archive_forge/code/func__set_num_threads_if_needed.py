import contextlib
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sized, Union
import torch
import torch.nn.functional as F
from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, Sampler
from typing_extensions import override
from lightning_fabric.utilities.cloud_io import _is_local_file_protocol
from lightning_fabric.utilities.data import _num_cpus_available
from lightning_fabric.utilities.rank_zero import rank_zero_info
from lightning_fabric.utilities.types import _PATH, ReduceOp
def _set_num_threads_if_needed(num_processes: int=1) -> None:
    if 'OMP_NUM_THREADS' not in os.environ:
        num_threads = _suggested_max_num_threads(num_processes)
        torch.set_num_threads(num_threads)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)