import asyncio
import enum
from functools import partial
import inspect
import logging
import traceback
from typing import Any, AsyncIterator, Generator, Generic, Optional, Tuple
import grpc
from grpc import _common
from grpc._cython import cygrpc
from . import _base_call
from ._metadata import Metadata
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadatumType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseType
from ._typing import SerializingFunction
def _create_rpc_error(initial_metadata: Metadata, status: cygrpc.AioRpcStatus) -> AioRpcError:
    return AioRpcError(_common.CYGRPC_STATUS_CODE_TO_STATUS_CODE[status.code()], Metadata.from_tuple(initial_metadata), Metadata.from_tuple(status.trailing_metadata()), details=status.details(), debug_error_string=status.debug_error_string())