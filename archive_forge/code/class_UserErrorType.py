import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import cast, NoReturn, Optional
import torch._guards
from . import config
from .config import is_fbcode
from .utils import counters
import logging
class UserErrorType(Enum):
    DYNAMIC_CONTROL_FLOW = auto()
    ANTI_PATTERN = auto()
    STANDARD_LIBRARY = auto()
    CONSTRAINT_VIOLATION = auto()
    DYNAMIC_DIM = auto()
    INVALID_INPUT = auto()