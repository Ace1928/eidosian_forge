from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
class SeriesSpec(Spec):
    """
    Parameters
    ----------
    data
    appendbcst : bool
    appendfcst : bool
    comptype
    compwt
    decimals
    modelspan
    name
    period
    precision
    to_print
    to_save
    span
    start
    title
    type

    Notes
    -----
    Rarely used arguments

    divpower
    missingcode
    missingval
    saveprecision
    trimzero
    """

    def __init__(self, data, name='Unnamed Series', appendbcst=False, appendfcst=False, comptype=None, compwt=1, decimals=0, modelspan=(), period=12, precision=0, to_print=[], to_save=[], span=(), start=(1, 1), title='', series_type=None, divpower=None, missingcode=-99999, missingval=1000000000):
        appendbcst, appendfcst = map(_bool_to_yes_no, [appendbcst, appendfcst])
        series_name = f'"{name[:64]}"'
        title = f'"{title[:79]}"'
        self.set_options(data=data, appendbcst=appendbcst, appendfcst=appendfcst, period=period, start=start, title=title, name=series_name)