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
def _append_to_avro_file(test_rows1, test_rows2, test_schema, test_expect=None, print_tables=True):
    _show__expect_rows(test_rows1, print_tables)
    _show__expect_rows(test_rows2, print_tables)
    test_filename = _get_tempfile_path()
    toavro(test_rows1, test_filename, schema=test_schema)
    appendavro(test_rows2, test_filename, schema=test_schema)
    test_actual = fromavro(test_filename)
    if test_expect is not None:
        test_expect2 = test_expect
    else:
        test_expect2 = cat(test_rows1, test_rows2)
    _assert_rows_are_equals(test_expect2, test_actual, print_tables)