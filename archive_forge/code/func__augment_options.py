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
def _augment_options(base_options: Sequence[ChannelArgumentType], compression: Optional[grpc.Compression]) -> Sequence[ChannelArgumentType]:
    compression_option = _compression.create_channel_option(compression)
    return tuple(base_options) + compression_option