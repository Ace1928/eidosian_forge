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
class HandshakeFragment:
    msg_type: int
    msg_len: int
    msg_seq: int
    frag_offset: int
    frag_len: int
    frag: bytes = attrs.field(repr=to_hex)