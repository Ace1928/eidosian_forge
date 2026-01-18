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
def _regular_table_line(values, col_widths):
    values_with_space = [f'| {v}' + ' ' * (w - len(v) + 1) for v, w in zip(values, col_widths)]
    return ''.join(values_with_space) + '|\n'