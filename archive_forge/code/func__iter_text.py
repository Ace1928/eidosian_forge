import codecs
import functools
import json
import os
import traceback
from testtools.compat import _b
from testtools.content_type import ContentType, JSON, UTF8_TEXT
def _iter_text(self):
    """Worker for iter_text - does the decoding."""
    encoding = self.content_type.parameters.get('charset', 'ISO-8859-1')
    decoder = codecs.getincrementaldecoder(encoding)()
    for bytes in self.iter_bytes():
        yield decoder.decode(bytes)
    final = decoder.decode(_b(''), True)
    if final:
        yield final