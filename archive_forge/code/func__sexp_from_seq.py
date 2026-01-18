import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
@staticmethod
def _sexp_from_seq(seq, tz_info_getter, isodatetime_columns):
    """ return a POSIXct vector from a sequence of time.struct_time
        elements. """
    tz_count = 0
    tz_info = None
    for elt in conversion.noconversion(seq):
        tmp = tz_info_getter(elt)
        if tz_info is None:
            tz_info = tmp
            tz_count = 1
        elif tz_info == tmp:
            tz_count += 1
        else:
            raise ValueError('Sequences of dates with different time zones not yet allowed.')
    if tz_info is None:
        tz_info = default_timezone if default_timezone else ''
    d = isodatetime_columns(seq)
    sexp = POSIXct._ISOdatetime(*d, tz=StrSexpVector((str(tz_info),)))
    return sexp