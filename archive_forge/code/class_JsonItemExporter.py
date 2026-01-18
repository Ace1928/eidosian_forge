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
class JsonItemExporter(BaseItemExporter):

    def __init__(self, file, **kwargs):
        super().__init__(dont_fail=True, **kwargs)
        self.file = file
        json_indent = self.indent if self.indent is not None and self.indent > 0 else None
        self._kwargs.setdefault('indent', json_indent)
        self._kwargs.setdefault('ensure_ascii', not self.encoding)
        self.encoder = ScrapyJSONEncoder(**self._kwargs)
        self.first_item = True

    def _beautify_newline(self):
        if self.indent is not None:
            self.file.write(b'\n')

    def _add_comma_after_first(self):
        if self.first_item:
            self.first_item = False
        else:
            self.file.write(b',')
            self._beautify_newline()

    def start_exporting(self):
        self.file.write(b'[')
        self._beautify_newline()

    def finish_exporting(self):
        self._beautify_newline()
        self.file.write(b']')

    def export_item(self, item):
        itemdict = dict(self._get_serialized_fields(item))
        data = to_bytes(self.encoder.encode(itemdict), self.encoding)
        self._add_comma_after_first()
        self.file.write(data)