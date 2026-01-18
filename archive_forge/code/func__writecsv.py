import io
import csv
import logging
from petl.util.base import Table, data
def _writecsv(table, source, mode, write_header, encoding, errors, **csvargs):
    rows = table if write_header else data(table)
    with source.open(mode) as buf:
        csvfile = io.TextIOWrapper(buf, encoding=encoding, errors=errors, newline='')
        try:
            writer = csv.writer(csvfile, **csvargs)
            for row in rows:
                writer.writerow(row)
            csvfile.flush()
        finally:
            csvfile.detach()