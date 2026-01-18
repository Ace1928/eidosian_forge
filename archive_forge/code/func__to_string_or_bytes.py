from __future__ import with_statement
from __future__ import unicode_literals
import io
import pybtex.io
from pybtex.plugin import Plugin
def _to_string_or_bytes(self, bib_data):
    stream = io.StringIO() if self.unicode_io else io.BytesIO()
    self.write_stream(bib_data, stream)
    return stream.getvalue()