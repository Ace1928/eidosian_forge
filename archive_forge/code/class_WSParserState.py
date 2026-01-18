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
class WSParserState(IntEnum):
    READ_HEADER = 1
    READ_PAYLOAD_LENGTH = 2
    READ_PAYLOAD_MASK = 3
    READ_PAYLOAD = 4