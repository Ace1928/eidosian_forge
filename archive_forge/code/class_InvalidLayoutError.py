from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
class InvalidLayoutError(Exception):
    pass