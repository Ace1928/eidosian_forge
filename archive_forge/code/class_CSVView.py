import io
import csv
import logging
from petl.util.base import Table, data
class CSVView(Table):

    def __init__(self, source, encoding, errors, header, **csvargs):
        self.source = source
        self.encoding = encoding
        self.errors = errors
        self.csvargs = csvargs
        self.header = header

    def __iter__(self):
        if self.header is not None:
            yield tuple(self.header)
        with self.source.open('rb') as buf:
            csvfile = io.TextIOWrapper(buf, encoding=self.encoding, errors=self.errors, newline='')
            try:
                reader = csv.reader(csvfile, **self.csvargs)
                for row in reader:
                    yield tuple(row)
            finally:
                csvfile.detach()