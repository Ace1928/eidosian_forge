from concurrent.futures import Executor
from typing import Any, Optional, Sequence
import grpc
from grpc import _common
from grpc import _compression
from grpc._cython import cygrpc
from . import _base_server
from ._interceptor import ServerInterceptor
from ._typing import ChannelArgumentType
Schedules a graceful shutdown in current event loop.

        The Cython AioServer doesn't hold a ref-count to this class. It should
        be safe to slightly extend the underlying Cython object's life span.
        