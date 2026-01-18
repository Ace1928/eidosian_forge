from __future__ import annotations
import contextlib
import hashlib
import inspect
from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Protocol, TypeVar, overload
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.time_util import time_to_seconds
Remove all fragments saved in this FragmentStorage.