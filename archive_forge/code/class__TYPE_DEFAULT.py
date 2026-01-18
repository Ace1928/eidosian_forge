from __future__ import annotations
import time
import typing
from enum import Enum
from socket import getdefaulttimeout
from ..exceptions import TimeoutStateError
class _TYPE_DEFAULT(Enum):
    token = -1