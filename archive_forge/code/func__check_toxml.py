from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def _check_toxml(table, expected, check=(), dump=None, **kwargs):
    with NamedTemporaryFile(delete=True, suffix='.xml') as f:
        filename = f.name
    toxml(table, filename, **kwargs)
    _dump_file(filename, dump)
    if check:
        try:
            actual = fromxml(filename, *check)
            _compare(expected, actual, dump)
        except Exception as ex:
            _dump_file(filename, False)
            raise ex