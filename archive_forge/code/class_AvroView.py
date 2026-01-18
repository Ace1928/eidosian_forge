from __future__ import absolute_import, division, print_function
import sys
import math
from collections import OrderedDict
from datetime import datetime, date, time
from decimal import Decimal
from petl.compat import izip_longest, text_type, string_types, PY3
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.transform.headers import skip, setheader
from petl.util.base import Table, dicts, fieldnames, iterpeek, wrap
class AvroView(Table):
    """Read rows from avro file with their types and logical types"""

    def __init__(self, source, limit, skips, **avro_args):
        self.source = source
        self.limit = limit
        self.skip = skips
        self.avro_args = avro_args
        self.avro_schema = None

    def get_avro_schema(self):
        """gets the schema stored in avro file header"""
        return self.avro_schema

    def __iter__(self):
        with self.source.open('rb') as source_file:
            avro_reader = self._open_reader(source_file)
            header = self._decode_schema(avro_reader)
            yield header
            for row in self._read_rows_from(avro_reader, header):
                yield row

    def _open_reader(self, source_file):
        """This could raise a error when the file is corrupt or is not avro"""
        import fastavro
        avro_reader = fastavro.reader(source_file, **self.avro_args)
        return avro_reader

    def _decode_schema(self, avro_reader):
        """extract the header from schema stored in avro file header"""
        self.avro_schema = avro_reader.writer_schema
        if self.avro_schema is None:
            return (None, None)
        schema_fields = self.avro_schema['fields']
        header = tuple((col['name'] for col in schema_fields))
        return header

    def _read_rows_from(self, avro_reader, header):
        count = 0
        maximum = self.limit if self.limit is not None else sys.maxsize
        for i, record in enumerate(avro_reader):
            if i < self.skip:
                continue
            if count >= maximum:
                break
            count += 1
            row = self._map_row_from(header, record)
            yield row

    def _map_row_from(self, header, record):
        """
        fastavro auto converts logical types defined in avro schema to 
        correspoding python types. E.g: 
        - avro type: long logicalType: timestamp-millis -> python datetime
        - avro type: int logicalType: date              -> python date
        - avro type: bytes logicalType: decimal         -> python Decimal
        """
        if header is None or PY3:
            r = tuple(record.values())
        else:
            r = tuple((record.get(col) for col in header))
        return r