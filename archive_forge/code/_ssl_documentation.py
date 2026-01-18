from __future__ import annotations
import contextlib
import operator as _operator
import ssl as _stdlib_ssl
from enum import Enum as _Enum
from typing import TYPE_CHECKING, Any, ClassVar, Final as TFinal, Generic, TypeVar
import trio
from . import _sync
from ._highlevel_generic import aclose_forcefully
from ._util import ConflictDetector, final
from .abc import Listener, Stream
Close the transport listener.