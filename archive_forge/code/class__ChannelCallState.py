import copy
import functools
import logging
import os
import sys
import threading
import time
import types
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _grpcio_metadata  # pytype: disable=pyi-error
from grpc import _observability  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import IntegratedCallFactory
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import UserTag
import grpc.experimental  # pytype: disable=pyi-error
class _ChannelCallState(object):
    channel: cygrpc.Channel
    managed_calls: int
    threading: bool

    def __init__(self, channel: cygrpc.Channel):
        self.lock = threading.Lock()
        self.channel = channel
        self.managed_calls = 0
        self.threading = False

    def reset_postfork_child(self) -> None:
        self.managed_calls = 0

    def __del__(self):
        try:
            self.channel.close(cygrpc.StatusCode.cancelled, 'Channel deallocated!')
        except (TypeError, AttributeError):
            pass