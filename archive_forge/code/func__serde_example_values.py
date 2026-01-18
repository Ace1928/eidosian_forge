from __future__ import annotations
import atexit
import concurrent.futures
import datetime
import functools
import inspect
import logging
import threading
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, overload
import orjson
from typing_extensions import TypedDict
from langsmith import client as ls_client
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _serde_example_values(values: VT) -> VT:
    if values is None:
        return values
    bts = ls_client._dumps_json(values, serialize_py=False)
    return orjson.loads(bts)