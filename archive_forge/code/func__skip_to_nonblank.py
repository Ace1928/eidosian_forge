import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def _skip_to_nonblank(f, spacegroup, setting):
    """Read lines from f until a nonblank line not starting with a
    hash (#) is encountered and returns this and the next line."""
    while True:
        line1 = f.readline()
        if not line1:
            raise SpacegroupNotFoundError('invalid spacegroup %s, setting %i not found in data base' % (spacegroup, setting))
        line1.strip()
        if line1 and (not line1.startswith('#')):
            line2 = f.readline()
            break
    return (line1, line2)