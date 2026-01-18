import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
Write string literal value with a best effort attempt to avoid backslashes.