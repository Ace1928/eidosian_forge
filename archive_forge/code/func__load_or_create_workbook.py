from __future__ import absolute_import, print_function, division
import itertools
from petl.compat import PY3, text_type
from petl.util.base import Table, data
from petl.io.sources import read_source_from_arg, write_source_from_arg
def _load_or_create_workbook(filename, mode, sheet):
    if PY3:
        FileNotFound = FileNotFoundError
    else:
        FileNotFound = IOError
    import openpyxl
    wb = None
    if not (mode == 'overwrite' or (mode == 'replace' and sheet is None)):
        try:
            source = read_source_from_arg(filename)
            with source.open('rb') as source2:
                wb = openpyxl.load_workbook(filename=source2, read_only=False)
        except FileNotFound:
            wb = None
    if wb is None:
        wb = openpyxl.Workbook(write_only=True)
    return wb