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
def _beautify_indent(self, depth=1):
    if self.indent:
        self.xg.characters(' ' * self.indent * depth)