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
class _ServeCustomEncoders:
    """Group of custom encoders for common types that's not handled by FastAPI."""

    @staticmethod
    def encode_np_array(obj):
        assert isinstance(obj, np.ndarray)
        if obj.dtype.kind == 'f':
            obj = obj.astype(float)
        if obj.dtype.kind in {'i', 'u'}:
            obj = obj.astype(int)
        return obj.tolist()

    @staticmethod
    def encode_np_scaler(obj):
        assert isinstance(obj, np.generic)
        return obj.item()

    @staticmethod
    def encode_exception(obj):
        assert isinstance(obj, Exception)
        return str(obj)

    @staticmethod
    def encode_pandas_dataframe(obj):
        assert isinstance(obj, pd.DataFrame)
        return obj.to_dict(orient='records')