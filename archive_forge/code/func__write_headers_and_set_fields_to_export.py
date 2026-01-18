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