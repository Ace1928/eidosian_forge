from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def _prepare_self(self, inlined=None):
    if not self._prepared:
        self._stream = list(self._prepare(self._stream, inlined))
        self._prepared = True