import copy
import dataclasses
import json
import os
import posixpath
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
import fsspec
from huggingface_hub import DatasetCard, DatasetCardData
from . import config
from .features import Features, Value
from .splits import SplitDict
from .tasks import TaskTemplate, task_template_from_dict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values
def _dump_info(self, file, pretty_print=False):
    """Dump info in `file` file-like object open in bytes mode (to support remote files)"""
    file.write(json.dumps(asdict(self), indent=4 if pretty_print else None).encode('utf-8'))