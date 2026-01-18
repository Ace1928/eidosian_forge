import asyncio
import functools
import json
import random
import re
import sys
import zlib
from enum import IntEnum
from struct import Struct
from typing import (
from .base_protocol import BaseProtocol
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .helpers import NO_EXTENSIONS
from .streams import DataQueue
class WSMessage(NamedTuple):
    type: WSMsgType
    data: Any
    extra: Optional[str]

    def json(self, *, loads: Callable[[Any], Any]=json.loads) -> Any:
        """Return parsed JSON data.

        .. versionadded:: 0.22
        """
        return loads(self.data)