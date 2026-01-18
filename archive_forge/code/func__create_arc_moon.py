import numpy as np
from pygsp import utils
from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
def _create_arc_moon(self, N, sigmad, d, number, seed):
    rs = np.random.RandomState(seed)
    phi = rs.rand(N, 1) * np.pi
    r = 1
    rb = sigmad * rs.normal(size=(N, 1))
    ab = rs.rand(N, 1) * 2 * np.pi
    b = rb * np.exp(1j * ab)
    bx = np.real(b)
    by = np.imag(b)
    if number == 1:
        moonx = np.cos(phi) * r + bx + 0.5
        moony = -np.sin(phi) * r + by - (d - 1) / 2.0
    elif number == 2:
        moonx = np.cos(phi) * r + bx - 0.5
        moony = np.sin(phi) * r + by + (d - 1) / 2.0
    return np.concatenate((moonx, moony), axis=1)