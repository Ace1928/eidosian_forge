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
def _write_read_into_url(base_url):
    _write_read_file_into_url(base_url, 'filename10.csv')
    _write_read_file_into_url(base_url, 'filename11.csv', 'gz')
    _write_read_file_into_url(base_url, 'filename12.csv', 'xz')
    _write_read_file_into_url(base_url, 'filename13.csv', 'zst')
    _write_read_file_into_url(base_url, 'filename14.csv', 'lz4')
    _write_read_file_into_url(base_url, 'filename15.csv', 'snappy')
    _write_read_file_into_url(base_url, 'filename20.json')
    _write_read_file_into_url(base_url, 'filename21.json', 'gz')
    _write_read_file_into_url(base_url, 'filename30.avro', pkg='fastavro')
    _write_read_file_into_url(base_url, 'filename40.xlsx', pkg='openpyxl')
    _write_read_file_into_url(base_url, 'filename50.xls', pkg='xlwt')