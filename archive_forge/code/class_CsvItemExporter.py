import csv
import io
import marshal
import pickle
import pprint
from collections.abc import Mapping
from xml.sax.saxutils import XMLGenerator
from itemadapter import ItemAdapter, is_item
from scrapy.item import Item
from scrapy.utils.python import is_listlike, to_bytes, to_unicode
from scrapy.utils.serialize import ScrapyJSONEncoder
class CsvItemExporter(BaseItemExporter):

    def __init__(self, file, include_headers_line=True, join_multivalued=',', errors=None, **kwargs):
        super().__init__(dont_fail=True, **kwargs)
        if not self.encoding:
            self.encoding = 'utf-8'
        self.include_headers_line = include_headers_line
        self.stream = io.TextIOWrapper(file, line_buffering=False, write_through=True, encoding=self.encoding, newline='', errors=errors)
        self.csv_writer = csv.writer(self.stream, **self._kwargs)
        self._headers_not_written = True
        self._join_multivalued = join_multivalued

    def serialize_field(self, field, name, value):
        serializer = field.get('serializer', self._join_if_needed)
        return serializer(value)

    def _join_if_needed(self, value):
        if isinstance(value, (list, tuple)):
            try:
                return self._join_multivalued.join(value)
            except TypeError:
                pass
        return value

    def export_item(self, item):
        if self._headers_not_written:
            self._headers_not_written = False
            self._write_headers_and_set_fields_to_export(item)
        fields = self._get_serialized_fields(item, default_value='', include_empty=True)
        values = list(self._build_row((x for _, x in fields)))
        self.csv_writer.writerow(values)

    def finish_exporting(self):
        self.stream.detach()

    def _build_row(self, values):
        for s in values:
            try:
                yield to_unicode(s, self.encoding)
            except TypeError:
                yield s

    def _write_headers_and_set_fields_to_export(self, item):
        if self.include_headers_line:
            if not self.fields_to_export:
                self.fields_to_export = ItemAdapter(item).field_names()
            if isinstance(self.fields_to_export, Mapping):
                fields = self.fields_to_export.values()
            else:
                fields = self.fields_to_export
            row = list(self._build_row(fields))
            self.csv_writer.writerow(row)