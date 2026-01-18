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
def _ensure_receive_loop(self) -> None:
    if not self._receive_loop_spawned:
        trio.lowlevel.spawn_system_task(dtls_receive_loop, weakref.ref(self), self.socket)
        self._receive_loop_spawned = True