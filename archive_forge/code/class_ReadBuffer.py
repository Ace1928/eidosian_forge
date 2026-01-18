from __future__ import annotations
from collections.abc import (
from datetime import (
from os import PathLike
import sys
from typing import (
import numpy as np
class ReadBuffer(BaseBuffer, Protocol[AnyStr_co]):

    def read(self, __n: int=...) -> AnyStr_co:
        ...