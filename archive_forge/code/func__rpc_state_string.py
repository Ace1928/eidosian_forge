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
def _rpc_state_string(class_name: str, rpc_state: _RPCState) -> str:
    """Calculates error string for RPC."""
    with rpc_state.condition:
        if rpc_state.code is None:
            return '<{} object>'.format(class_name)
        elif rpc_state.code is grpc.StatusCode.OK:
            return _OK_RENDEZVOUS_REPR_FORMAT.format(class_name, rpc_state.code, rpc_state.details)
        else:
            return _NON_OK_RENDEZVOUS_REPR_FORMAT.format(class_name, rpc_state.code, rpc_state.details, rpc_state.debug_error_string)