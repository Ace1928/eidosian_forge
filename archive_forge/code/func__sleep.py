from __future__ import annotations
import time
from typing import TYPE_CHECKING
import anyio
def _sleep(self, seconds: float) -> None:
    time.sleep(seconds)