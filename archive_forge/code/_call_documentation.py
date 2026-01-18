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
This method prepares the RPC for receiving/sending messages.

        All other operations around the stream should only happen after the
        completion of this method.
        