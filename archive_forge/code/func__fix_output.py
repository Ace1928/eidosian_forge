import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _fix_output(output, usemask=True, asrecarray=False):
    """
    Private function: return a recarray, a ndarray, a MaskedArray
    or a MaskedRecords depending on the input parameters
    """
    if not isinstance(output, MaskedArray):
        usemask = False
    if usemask:
        if asrecarray:
            output = output.view(MaskedRecords)
    else:
        output = ma.filled(output)
        if asrecarray:
            output = output.view(recarray)
    return output