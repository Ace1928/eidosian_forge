import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _izip_records(seqarrays, fill_value=None, flatten=True):
    """
    Returns an iterator of concatenated items from a sequence of arrays.

    Parameters
    ----------
    seqarrays : sequence of arrays
        Sequence of arrays.
    fill_value : {None, integer}
        Value used to pad shorter iterables.
    flatten : {True, False},
        Whether to
    """
    if flatten:
        zipfunc = _izip_fields_flat
    else:
        zipfunc = _izip_fields
    for tup in itertools.zip_longest(*seqarrays, fillvalue=fill_value):
        yield tuple(zipfunc(tup))