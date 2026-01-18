import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def _human_readable_bytes(bytes: int) -> str:
    if bytes < 10 ** 3:
        return f'{bytes} Byte'
    elif bytes < 10 ** 6:
        return f'{bytes / 10 ** 3:.1f} kB'
    elif bytes < 10 ** 9:
        return f'{bytes / 10 ** 6:.1f} MB'
    else:
        return f'{bytes / 10 ** 9:.1f} GB'