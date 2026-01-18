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
def _utc(year, month, day, hour=0, minute=0, second=0, microsecond=0):
    u = datetime(year, month, day, hour, minute, second, microsecond)
    if PY3:
        return u.replace(tzinfo=timezone.utc)
    return u.replace(tzinfo=pytz.utc)