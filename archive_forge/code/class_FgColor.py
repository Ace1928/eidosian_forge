import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
class FgColor(AnsiSequence):
    """Base class for ANSI Sequences which set foreground text color"""
    pass