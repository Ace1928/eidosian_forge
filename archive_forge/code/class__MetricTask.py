import asyncio
import copy
import importlib
import inspect
import logging
import math
import os
import random
import string
import threading
import time
import traceback
from abc import ABC, abstractmethod
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import requests
import ray
import ray.util.serialization_addons
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.utils import import_attr
from ray._private.worker import LOCAL_MODE, SCRIPT_MODE
from ray._raylet import MessagePackSerializer
from ray.actor import ActorHandle
from ray.exceptions import RayTaskError
from ray.serve._private.constants import HTTP_PROXY_TIMEOUT, SERVE_LOGGER_NAME
from ray.types import ObjectRef
from ray.util.serialization import StandaloneSerializationContext
class _MetricTask:

    def __init__(self, task_func, interval_s, callback_func):
        """
        Args:
            task_func: a callable that MetricsPusher will try to call in each loop.
            interval_s: the interval of each task_func is supposed to be called.
            callback_func: callback function is called when task_func is done, and
                the result of task_func is passed to callback_func as the first
                argument, and the timestamp of the call is passed as the second
                argument.
        """
        self.task_func: Callable = task_func
        self.interval_s: float = interval_s
        self.callback_func: Callable[[Any, float]] = callback_func
        self.last_ref: Optional[ray.ObjectRef] = None
        self.last_call_succeeded_time: Optional[float] = time.time()