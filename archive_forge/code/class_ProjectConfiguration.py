import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class ProjectConfiguration:
    """
    Configuration for the Accelerator object based on inner-project needs.
    """
    project_dir: str = field(default=None, metadata={'help': 'A path to a directory for storing data.'})
    logging_dir: str = field(default=None, metadata={'help': 'A path to a directory for storing logs of locally-compatible loggers. If None, defaults to `project_dir`.'})
    automatic_checkpoint_naming: bool = field(default=False, metadata={'help': 'Whether saved states should be automatically iteratively named.'})
    total_limit: int = field(default=None, metadata={'help': 'The maximum number of total saved states to keep.'})
    iteration: int = field(default=0, metadata={'help': 'The current save iteration.'})
    save_on_each_node: bool = field(default=False, metadata={'help': 'When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one'})

    def set_directories(self, project_dir: str=None):
        """Sets `self.project_dir` and `self.logging_dir` to the appropriate values."""
        self.project_dir = project_dir
        if self.logging_dir is None:
            self.logging_dir = project_dir

    def __post_init__(self):
        self.set_directories(self.project_dir)