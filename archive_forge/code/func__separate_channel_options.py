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
def _separate_channel_options(options: Sequence[ChannelArgumentType]) -> Tuple[Sequence[ChannelArgumentType], Sequence[ChannelArgumentType]]:
    """Separates core channel options from Python channel options."""
    core_options = []
    python_options = []
    for pair in options:
        if pair[0] == grpc.experimental.ChannelOptions.SingleThreadedUnaryStream:
            python_options.append(pair)
        else:
            core_options.append(pair)
    return (python_options, core_options)