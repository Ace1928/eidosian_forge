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
@functools.lru_cache
def _xor_table() -> List[bytes]:
    return [bytes((a ^ b for a in range(256))) for b in range(256)]