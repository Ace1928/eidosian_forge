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
def _deliveries(state: _ChannelConnectivityState) -> List[Callable[[grpc.ChannelConnectivity], None]]:
    callbacks_needing_update = []
    for callback_and_connectivity in state.callbacks_and_connectivities:
        callback, callback_connectivity = callback_and_connectivity
        if callback_connectivity is not state.connectivity:
            callbacks_needing_update.append(callback)
            callback_and_connectivity[1] = state.connectivity
    return callbacks_needing_update