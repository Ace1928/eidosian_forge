from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
def _get_overlap_public_nat_methods(klass, as_tuple=False):
    """
    Get overlapping public methods between NaT and another class.

    Parameters
    ----------
    klass : type
        The class to compare with NaT
    as_tuple : bool, default False
        Whether to return a list of tuples of the form (klass, method).

    Returns
    -------
    overlap : list
    """
    nat_names = dir(NaT)
    klass_names = dir(klass)
    overlap = [x for x in nat_names if x in klass_names and (not x.startswith('_')) and callable(getattr(klass, x))]
    if klass is Timedelta:
        ts_names = dir(Timestamp)
        overlap = [x for x in overlap if x not in ts_names]
    if as_tuple:
        overlap = [(klass, method) for method in overlap]
    overlap.sort()
    return overlap