import math
import cupy
from cupy.linalg import _util
@cupy.fuse
def _expm_inner(E, A, A2, A4, A6, b):
    u1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    u2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * E
    v1 = b[12] * A6 + b[10] * A4 + b[8] * A
    v2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * E
    return (u1, u2, v1, v2)