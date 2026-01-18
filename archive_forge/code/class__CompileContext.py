import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
class _CompileContext:

    def __init__(self, uri, filename, default_filters, buffer_filters, imports, future_imports, source_encoding, generate_magic_comment, strict_undefined, enable_loop, reserved_names):
        self.uri = uri
        self.filename = filename
        self.default_filters = default_filters
        self.buffer_filters = buffer_filters
        self.imports = imports
        self.future_imports = future_imports
        self.source_encoding = source_encoding
        self.generate_magic_comment = generate_magic_comment
        self.strict_undefined = strict_undefined
        self.enable_loop = enable_loop
        self.reserved_names = reserved_names