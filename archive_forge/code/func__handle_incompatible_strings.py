import itertools
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import mlflow
from packaging.version import Version  # type: ignore
import wandb
from wandb import Artifact
from .internals import internal
from .internals.util import Namespace, for_each
@staticmethod
def _handle_incompatible_strings(s: str) -> str:
    valid_chars = '[^a-zA-Z0-9_\\-\\.]'
    replacement = '__'
    return re.sub(valid_chars, replacement, s)