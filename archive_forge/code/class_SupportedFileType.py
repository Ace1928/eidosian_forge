import argparse
from dataclasses import dataclass
from enum import Enum
import os.path
import tempfile
import typer
from typing import Optional
import requests
from ray.tune.experiment.config_parser import _make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
class SupportedFileType(str, Enum):
    """Supported file types for RLlib, used for CLI argument validation."""
    yaml = 'yaml'
    python = 'python'