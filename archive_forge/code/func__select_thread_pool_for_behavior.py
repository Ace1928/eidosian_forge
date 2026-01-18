from __future__ import annotations
import collections
from concurrent import futures
import contextvars
import enum
import logging
import threading
import time
import traceback
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _interceptor  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ArityAgnosticMethodHandler
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import ServerCallbackTag
from grpc._typing import ServerTagCallbackType
def _select_thread_pool_for_behavior(behavior: ArityAgnosticMethodHandler, default_thread_pool: futures.ThreadPoolExecutor) -> futures.ThreadPoolExecutor:
    if hasattr(behavior, 'experimental_thread_pool') and isinstance(behavior.experimental_thread_pool, futures.ThreadPoolExecutor):
        return behavior.experimental_thread_pool
    else:
        return default_thread_pool