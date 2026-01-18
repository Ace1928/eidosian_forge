import numpy as np
from scipy import ndimage as ndi
from .footprints import _footprint_is_sequence, pad_footprint
from .misc import default_footprint
def _iterate_binary_func(binary_func, image, footprint, out, border_value):
    """Helper to call `binary_func` for each footprint in a sequence.

    binary_func is a binary morphology function that accepts "structure",
    "output" and "iterations" keyword arguments
    (e.g. `scipy.ndimage.binary_erosion`).
    """
    fp, num_iter = footprint[0]
    binary_func(image, structure=fp, output=out, iterations=num_iter, border_value=border_value)
    for fp, num_iter in footprint[1:]:
        binary_func(out.copy(), structure=fp, output=out, iterations=num_iter, border_value=border_value)
    return out