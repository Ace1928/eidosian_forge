from __future__ import annotations
from collections.abc import (
from datetime import (
from os import PathLike
import sys
from typing import (
import numpy as np
class ReadPickleBuffer(ReadBuffer[bytes], Protocol):

    def readline(self) -> bytes:
        ...