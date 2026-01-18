import cupy
from cupyx.scipy.interpolate._interpolate import PPoly
@staticmethod
def _edge_case(h0, h1, m0, m1):
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    mask = cupy.sign(d) != cupy.sign(m0)
    mask2 = (cupy.sign(m0) != cupy.sign(m1)) & (cupy.abs(d) > 3.0 * cupy.abs(m0))
    mmm = ~mask & mask2
    d[mask] = 0.0
    d[mmm] = 3.0 * m0[mmm]
    return d