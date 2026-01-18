from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def _init_loader(self):
    if self.loader is None:
        from genshi.template.loader import TemplateLoader
        if self.filename:
            if self.filepath != self.filename:
                basedir = os.path.normpath(self.filepath)[:-len(os.path.normpath(self.filename))]
            else:
                basedir = os.path.dirname(self.filename)
        else:
            basedir = '.'
        self.loader = TemplateLoader([os.path.abspath(basedir)])