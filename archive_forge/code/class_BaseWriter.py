from __future__ import with_statement
from __future__ import unicode_literals
import io
import pybtex.io
from pybtex.plugin import Plugin
class BaseWriter(Plugin):
    unicode_io = False

    def __init__(self, encoding=None):
        self.encoding = encoding or pybtex.io.get_default_encoding()

    def write_file(self, bib_data, filename):
        open_file = pybtex.io.open_unicode if self.unicode_io else pybtex.io.open_raw
        mode = 'w' if self.unicode_io else 'wb'
        with open_file(filename, mode, encoding=self.encoding) as stream:
            self.write_stream(bib_data, stream)
            if hasattr(stream, 'getvalue'):
                return stream.getvalue()

    def write_stream(self, bib_data, stream):
        raise NotImplementedError

    def _to_string_or_bytes(self, bib_data):
        stream = io.StringIO() if self.unicode_io else io.BytesIO()
        self.write_stream(bib_data, stream)
        return stream.getvalue()

    def to_string(self, bib_data):
        result = self._to_string_or_bytes(bib_data)
        return result if self.unicode_io else result.decode(self.encoding)

    def to_bytes(self, bib_data):
        result = self._to_string_or_bytes(bib_data)
        return result.encode(self.encoding) if self.unicode_io else result