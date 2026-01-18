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
def _create_avro_example(test_schema, test_table):
    parsed_schema = fastavro.parse_schema(test_schema)
    rows = dicts(test_table)
    with NamedTemporaryFile(delete=False, mode='wb') as fo:
        fastavro.writer(fo, parsed_schema, rows)
        return fo.name