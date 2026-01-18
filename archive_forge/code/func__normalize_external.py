from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
def _normalize_external(external):
    """ Normalize external into a well-formed list of tuples and return. """
    if external is None:
        return []
    try:
        return [_external_entry((external, 0, h5f.UNLIMITED))]
    except TypeError:
        pass
    return [_external_entry(entry) for entry in external]