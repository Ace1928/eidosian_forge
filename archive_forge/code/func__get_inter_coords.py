import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _get_inter_coords(coords, strand=1):
    """Return list of pairs covering intervening ranges (PRIVATE).

    From the given pairs of coordinates, returns a list of pairs
    covering the intervening ranges.
    """
    if strand == -1:
        sorted_coords = [(max(a, b), min(a, b)) for a, b in coords]
        inter_coords = list(chain(*sorted_coords))[1:-1]
        return list(zip(inter_coords[1::2], inter_coords[::2]))
    else:
        inter_coords = list(chain(*coords))[1:-1]
        return list(zip(inter_coords[::2], inter_coords[1::2]))