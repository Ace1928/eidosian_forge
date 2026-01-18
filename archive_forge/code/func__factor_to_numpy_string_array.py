import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
def _factor_to_numpy_string_array(obj):
    levels = obj.do_slot('levels')
    res = numpy.array(tuple((None if x is rinterface.NA_Character else levels[x - 1] for x in obj)))
    return res