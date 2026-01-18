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
def _all_to_all_api(*args, **kwargs):
    """Annotate the function with an indication that it's a all to all API, and that it
    is an operation that requires all inputs to be materialized in-memory to execute.
    """

    def wrap(obj):
        _insert_doc_at_pattern(obj, message='This operation requires all inputs to be materialized in object store for it to execute.', pattern='Examples:', insert_after=False, directive='note')
        return obj
    return wrap