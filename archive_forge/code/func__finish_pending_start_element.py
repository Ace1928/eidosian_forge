import os, urllib.parse, urllib.request
import io
import codecs
from . import handler
from . import xmlreader
def _finish_pending_start_element(self, endElement=False):
    if self._pending_start_element:
        self._write('>')
        self._pending_start_element = False