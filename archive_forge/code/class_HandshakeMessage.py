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
@attrs.frozen
class HandshakeMessage:
    record_version: bytes = attrs.field(repr=to_hex)
    msg_type: HandshakeType
    msg_seq: int
    body: bytearray = attrs.field(repr=to_hex)