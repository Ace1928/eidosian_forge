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
def _get_definition_from_record(prop, val, fprev, dprev, fill_missing):
    if fprev is None:
        fprev = OrderedDict()
    if dprev is None:
        dprev = OrderedDict()
    props = list(val.keys())
    row = list(val.values())
    _update_field_defs_from(props, row, fprev, dprev, fill_missing)
    schema_fields = [item for item in fprev.values()]
    tdef = {'type': 'record', 'name': prop + '_record', 'namespace': 'avro', 'fields': schema_fields}
    return (tdef, dprev, fprev)