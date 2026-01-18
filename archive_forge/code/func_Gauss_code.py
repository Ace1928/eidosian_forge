import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def Gauss_code(self):
    """
        Return a Gauss code for the link.  The Gauss code is computed
        from a DT code, so the Gauss code will use the same indexing
        of crossings as is used for the DT code.  Requires that all
        components be closed.
        """
    dt, sizes = self.DT_code(signed=False, return_sizes=True)
    if dt is None:
        return None
    evens = [y for x in dt for y in x]
    size = 2 * len(evens)
    counts = [None] * size
    for odd, N in zip(range(1, size, 2), evens):
        even = abs(N)
        if even < odd:
            counts[even - 1] = -N
            counts[odd - 1] = N
        else:
            O = odd if N > 0 else -odd
            counts[even - 1] = -O
            counts[odd - 1] = O
    gauss = []
    start = 0
    for size in sizes:
        end = start + size
        gauss.append(tuple(counts[start:end]))
        start = end
    return gauss