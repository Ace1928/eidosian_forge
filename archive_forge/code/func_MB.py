from __future__ import annotations
import io
from typing import Callable
from typing_extensions import override
def MB(i: int) -> int:
    return int(i // 1024 ** 2)