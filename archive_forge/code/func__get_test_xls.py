from __future__ import division, print_function, absolute_import
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xls import fromxls, toxls
from petl.test.helpers import ieq
def _get_test_xls():
    try:
        import pkg_resources
        return pkg_resources.resource_filename('petl', 'test/resources/test.xls')
    except:
        return None