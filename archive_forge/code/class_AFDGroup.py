from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
@attrs.define(eq=False)
class AFDGroup:
    size: int
    handle: Handle