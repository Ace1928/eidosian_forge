import struct
import typing as t
def FF(a: int, b: int, c: int, d: int, x: int, s: int) -> int:
    return ROTL(a + F(b, c, d) + x & 4294967295, s)