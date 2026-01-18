from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
def _check_replaced(self) -> None:
    if self._replaced:
        raise trio.BrokenResourceError('peer tore down this connection to start a new one')