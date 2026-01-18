from snappy.snap import t3mlite as t3m
from truncatedComplex import *
def _fixed_points(m):
    cinv = 1 / m[1, 0]
    p = (m[0, 0] - m[1, 1]) * cinv / 2
    d = p ** 2 + m[0, 1] * cinv
    s = d.sqrt()
    return [p - s, p + s]