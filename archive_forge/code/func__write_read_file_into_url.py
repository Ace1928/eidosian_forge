from __future__ import absolute_import, print_function, division
import sys
import os
from importlib import import_module
import pytest
from petl.compat import PY3
from petl.test.helpers import ieq, eq_
from petl.io.avro import fromavro, toavro
from petl.io.csv import fromcsv, tocsv
from petl.io.json import fromjson, tojson
from petl.io.xlsx import fromxlsx, toxlsx
from petl.io.xls import fromxls, toxls
from petl.util.vis import look
def _write_read_file_into_url(base_url, filename, compression=None, pkg=None):
    if not _is_installed(pkg, filename):
        return
    source_url = _build_source_url_from(base_url, filename, compression)
    if source_url is None:
        return
    actual = None
    if '.avro' in filename:
        toavro(_table, source_url)
        actual = fromavro(source_url)
    elif '.xlsx' in filename:
        toxlsx(_table, source_url, 'test1', mode='overwrite')
        toxlsx(_table2, source_url, 'test2', mode='add')
        actual = fromxlsx(source_url, 'test1')
    elif '.xls' in filename:
        toxls(_table, source_url, 'test')
        actual = fromxls(source_url, 'test')
    elif '.json' in filename:
        tojson(_table, source_url)
        actual = fromjson(source_url)
    elif '.csv' in filename:
        tocsv(_table, source_url, encoding='ascii', lineterminator='\n')
        actual = fromcsv(source_url, encoding='ascii')
    if actual is not None:
        _show__rows_from('Expected:', _table)
        _show__rows_from('Actual:', actual)
        ieq(_table, actual)
        ieq(_table, actual)
    else:
        print('\n    - %s SKIPPED ' % filename, file=sys.stderr, end='')