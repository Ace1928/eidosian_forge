from functools import partial
import numpy as np
def _ch_helper(gamma, s, r, h, p0, p1, x):
    """Helper function for generating picklable cubehelix colormaps."""
    xg = x ** gamma
    a = h * xg * (1 - xg) / 2
    phi = 2 * np.pi * (s / 3 + r * x)
    return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))