from __future__ import absolute_import, division, print_function
import sys
import math
from collections import OrderedDict
from datetime import datetime, date, time
from decimal import Decimal
from petl.compat import izip_longest, text_type, string_types, PY3
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.transform.headers import skip, setheader
from petl.util.base import Table, dicts, fieldnames, iterpeek, wrap
def _get_precision_from_decimal(curr, val, prev):
    if val is None:
        prec = scale = 0
    else:
        prec, scale, _, _ = precision_and_scale(val)
    if prev is not None:
        prec0, scale0 = (prev.get('precision'), prev.get('scale'))
        prec, scale = (max(prec, prec0), max(scale, scale0))
    prec = max(prec, 8)
    curr = {'precision': prec, 'scale': scale}
    return (curr, prec, scale)