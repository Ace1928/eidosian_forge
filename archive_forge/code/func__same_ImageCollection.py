import numpy as np
import skimage.io as io
from skimage._shared import testing
def _same_ImageCollection(collection1, collection2):
    """
    Ancillary function to compare two ImageCollection objects, checking that
    their constituent arrays are equal.
    """
    if len(collection1) != len(collection2):
        return False
    for ext1, ext2 in zip(collection1, collection2):
        if not np.all(ext1 == ext2):
            return False
    return True