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
class XmlItemExporter(BaseItemExporter):

    def __init__(self, file, **kwargs):
        self.item_element = kwargs.pop('item_element', 'item')
        self.root_element = kwargs.pop('root_element', 'items')
        super().__init__(**kwargs)
        if not self.encoding:
            self.encoding = 'utf-8'
        self.xg = XMLGenerator(file, encoding=self.encoding)

    def _beautify_newline(self, new_item=False):
        if self.indent is not None and (self.indent > 0 or new_item):
            self.xg.characters('\n')

    def _beautify_indent(self, depth=1):
        if self.indent:
            self.xg.characters(' ' * self.indent * depth)

    def start_exporting(self):
        self.xg.startDocument()
        self.xg.startElement(self.root_element, {})
        self._beautify_newline(new_item=True)

    def export_item(self, item):
        self._beautify_indent(depth=1)
        self.xg.startElement(self.item_element, {})
        self._beautify_newline()
        for name, value in self._get_serialized_fields(item, default_value=''):
            self._export_xml_field(name, value, depth=2)
        self._beautify_indent(depth=1)
        self.xg.endElement(self.item_element)
        self._beautify_newline(new_item=True)

    def finish_exporting(self):
        self.xg.endElement(self.root_element)
        self.xg.endDocument()

    def _export_xml_field(self, name, serialized_value, depth):
        self._beautify_indent(depth=depth)
        self.xg.startElement(name, {})
        if hasattr(serialized_value, 'items'):
            self._beautify_newline()
            for subname, value in serialized_value.items():
                self._export_xml_field(subname, value, depth=depth + 1)
            self._beautify_indent(depth=depth)
        elif is_listlike(serialized_value):
            self._beautify_newline()
            for value in serialized_value:
                self._export_xml_field('value', value, depth=depth + 1)
            self._beautify_indent(depth=depth)
        elif isinstance(serialized_value, str):
            self.xg.characters(serialized_value)
        else:
            self.xg.characters(str(serialized_value))
        self.xg.endElement(name)
        self._beautify_newline()