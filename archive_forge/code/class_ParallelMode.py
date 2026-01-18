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
class ParallelMode(Enum):
    NOT_PARALLEL = 'not_parallel'
    NOT_DISTRIBUTED = 'not_distributed'
    DISTRIBUTED = 'distributed'
    SAGEMAKER_MODEL_PARALLEL = 'sagemaker_model_parallel'
    SAGEMAKER_DATA_PARALLEL = 'sagemaker_data_parallel'
    TPU = 'tpu'