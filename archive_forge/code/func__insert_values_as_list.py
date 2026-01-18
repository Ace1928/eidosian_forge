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
def _insert_values_as_list(metadata, name, values):
    if values is None:
        return metadata
    if isinstance(values, str):
        values = [values]
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return metadata
    metadata[name] = values
    return metadata