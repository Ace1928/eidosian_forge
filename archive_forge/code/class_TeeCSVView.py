import io
import csv
import logging
from petl.util.base import Table, data
class TeeCSVView(Table):

    def __init__(self, table, source=None, encoding=None, errors='strict', write_header=True, **csvargs):
        self.table = table
        self.source = source
        self.write_header = write_header
        self.encoding = encoding
        self.errors = errors
        self.csvargs = csvargs

    def __iter__(self):
        with self.source.open('wb') as buf:
            csvfile = io.TextIOWrapper(buf, encoding=self.encoding, errors=self.errors, newline='')
            try:
                writer = csv.writer(csvfile, **self.csvargs)
                it = iter(self.table)
                try:
                    hdr = next(it)
                except StopIteration:
                    return
                if self.write_header:
                    writer.writerow(hdr)
                yield tuple(hdr)
                for row in it:
                    writer.writerow(row)
                    yield tuple(row)
                csvfile.flush()
            finally:
                csvfile.detach()