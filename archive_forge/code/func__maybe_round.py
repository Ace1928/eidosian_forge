import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
import yaml
from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError
from . import __version__
from .models.auto.modeling_auto import (
from .training_args import ParallelMode
from .utils import (
def _maybe_round(v, decimals=4):
    if isinstance(v, float) and len(str(v).split('.')) > 1 and (len(str(v).split('.')[1]) > decimals):
        return f'{v:.{decimals}f}'
    return str(v)