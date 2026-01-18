from __future__ import annotations
from typing import (
def _mklbl(prefix: str, n: int):
    return [f'{prefix}{i}' for i in range(n)]