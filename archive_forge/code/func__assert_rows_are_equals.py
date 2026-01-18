from __future__ import absolute_import, print_function, division
import math
from datetime import datetime, date
from decimal import Decimal
from tempfile import NamedTemporaryFile
import pytest
from petl.compat import PY3
from petl.transform.basics import cat
from petl.util.base import dicts
from petl.util.vis import look
from petl.test.helpers import ieq
from petl.io.avro import fromavro, toavro, appendavro
from petl.test.io.test_avro_schemas import schema0, schema1, schema2, \
def _assert_rows_are_equals(test_expect, test_actual, print_tables=True):
    if print_tables:
        _show__rows_from('Actual:', test_actual)
        avro_schema = test_actual.get_avro_schema()
        print('\nSchema:\n', avro_schema)
    ieq(test_expect, test_actual)
    ieq(test_expect, test_actual)