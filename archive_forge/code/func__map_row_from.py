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
def _map_row_from(self, header, record):
    """
        fastavro auto converts logical types defined in avro schema to 
        correspoding python types. E.g: 
        - avro type: long logicalType: timestamp-millis -> python datetime
        - avro type: int logicalType: date              -> python date
        - avro type: bytes logicalType: decimal         -> python Decimal
        """
    if header is None or PY3:
        r = tuple(record.values())
    else:
        r = tuple((record.get(col) for col in header))
    return r