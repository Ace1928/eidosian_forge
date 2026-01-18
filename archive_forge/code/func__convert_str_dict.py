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
def _convert_str_dict(passed_value: dict):
    """Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."""
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            if value.lower() in ('true', 'false'):
                passed_value[key] = value.lower() == 'true'
            elif value.isdigit():
                passed_value[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                passed_value[key] = float(value)
    return passed_value