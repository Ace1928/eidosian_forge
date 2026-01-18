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
def ConsumptionAPI(*args, **kwargs):
    """Annotate the function with an indication that it's a consumption API, and that it
    will trigger Dataset execution.
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return _consumption_api()(args[0])
    return _consumption_api(*args, **kwargs)