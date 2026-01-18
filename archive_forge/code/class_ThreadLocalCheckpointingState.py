from contextlib import contextmanager
from dataclasses import dataclass
import functools
import threading
from typing import Any, Dict, Generator, Optional, Tuple
import weakref
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from fairscale.internal.containers import pack_kwargs, split_non_tensors, unpack_kwargs, unpack_non_tensors
from .checkpoint_utils import patch_batchnorm
@dataclass
class ThreadLocalCheckpointingState(threading.local):
    is_checkpointing: bool = False
    is_recomputing: bool = False
    is_checkpointing_disabled: bool = False