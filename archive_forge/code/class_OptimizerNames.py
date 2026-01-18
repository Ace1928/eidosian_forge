import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    ADAMW_HF = 'adamw_hf'
    ADAMW_TORCH = 'adamw_torch'
    ADAMW_TORCH_FUSED = 'adamw_torch_fused'
    ADAMW_TORCH_XLA = 'adamw_torch_xla'
    ADAMW_TORCH_NPU_FUSED = 'adamw_torch_npu_fused'
    ADAMW_APEX_FUSED = 'adamw_apex_fused'
    ADAFACTOR = 'adafactor'
    ADAMW_ANYPRECISION = 'adamw_anyprecision'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    ADAMW_BNB = 'adamw_bnb_8bit'
    ADAMW_8BIT = 'adamw_8bit'
    LION_8BIT = 'lion_8bit'
    LION = 'lion_32bit'
    PAGED_ADAMW = 'paged_adamw_32bit'
    PAGED_ADAMW_8BIT = 'paged_adamw_8bit'
    PAGED_LION = 'paged_lion_32bit'
    PAGED_LION_8BIT = 'paged_lion_8bit'
    RMSPROP = 'rmsprop'
    RMSPROP_BNB = 'rmsprop_bnb'
    RMSPROP_8BIT = 'rmsprop_bnb_8bit'
    RMSPROP_32BIT = 'rmsprop_bnb_32bit'