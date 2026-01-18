import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def _truncated_repr(obj: Any) -> str:
    """Utility to return a truncated object representation for error messages."""
    msg = str(obj)
    if len(msg) > 200:
        msg = msg[:200] + '...'
    return msg