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
class _InitialMetadataFlags(int):
    """Stores immutable initial metadata flags"""

    def __new__(cls, value: int=_EMPTY_FLAGS):
        value &= cygrpc.InitialMetadataFlags.used_mask
        return super(_InitialMetadataFlags, cls).__new__(cls, value)

    def with_wait_for_ready(self, wait_for_ready: Optional[bool]) -> int:
        if wait_for_ready is not None:
            if wait_for_ready:
                return self.__class__(self | cygrpc.InitialMetadataFlags.wait_for_ready | cygrpc.InitialMetadataFlags.wait_for_ready_explicitly_set)
            elif not wait_for_ready:
                return self.__class__(self & ~cygrpc.InitialMetadataFlags.wait_for_ready | cygrpc.InitialMetadataFlags.wait_for_ready_explicitly_set)
        return self