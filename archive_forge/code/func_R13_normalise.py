from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def R13_normalise(v, sign=0):
    dot = r13_dot(v, v)
    if sign == 0:
        d = abs(dot)
    else:
        d = sign * dot
    denom = d.sqrt()
    return v / denom