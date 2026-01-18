import typing
from typing import Any, Optional
def _extgcd(a: int, b: int) -> tuple[int, int]:
    c, d = (a, b)
    x, u = (1, 0)
    while d:
        r = c // d
        c, d = (d, c - d * r)
        x, u = (u, x - u * r)
    return (c, x)