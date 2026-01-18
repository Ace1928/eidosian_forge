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
def _update_field_defs_from(props, row, fields, previous, fill_missing):
    for prop, val in izip_longest(props, row):
        if prop is None:
            break
        dprev = previous.get(prop + '_prec')
        fprev = previous.get(prop + '_prop')
        fcurr = None
        if isinstance(val, dict):
            tdef, dcurr, fcurr = _get_definition_from_record(prop, val, fprev, dprev, fill_missing)
        else:
            tdef, dcurr = _get_definition_from_type_of(prop, val, dprev)
        if tdef is not None:
            fields[prop] = {'name': prop, 'type': ['null', tdef]}
        elif fill_missing:
            fields[prop] = {'name': prop, 'type': ['null', 'string']}
        if dcurr is not None:
            previous[prop + '_prec'] = dcurr
        if fcurr is not None:
            previous[prop + '_prop'] = fcurr