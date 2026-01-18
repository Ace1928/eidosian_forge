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
class ComputeEnvironment(str, enum.Enum):
    """
    Represents a type of the compute environment.

    Values:

        - **LOCAL_MACHINE** -- private/custom cluster hardware.
        - **AMAZON_SAGEMAKER** -- Amazon SageMaker as compute environment.
    """
    LOCAL_MACHINE = 'LOCAL_MACHINE'
    AMAZON_SAGEMAKER = 'AMAZON_SAGEMAKER'