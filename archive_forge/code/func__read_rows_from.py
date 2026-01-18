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
def _read_rows_from(self, avro_reader, header):
    count = 0
    maximum = self.limit if self.limit is not None else sys.maxsize
    for i, record in enumerate(avro_reader):
        if i < self.skip:
            continue
        if count >= maximum:
            break
        count += 1
        row = self._map_row_from(header, record)
        yield row