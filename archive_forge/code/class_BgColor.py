import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
class BgColor(AnsiSequence):
    """Base class for ANSI Sequences which set background text color"""
    pass