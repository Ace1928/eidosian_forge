import json
import os
import re
import shutil
import stat
import tempfile
import types
import weakref
from mako import cache
from mako import codegen
from mako import compat
from mako import exceptions
from mako import runtime
from mako import util
from mako.lexer import Lexer
class DefTemplate(Template):
    """A :class:`.Template` which represents a callable def in a parent
    template."""

    def __init__(self, parent, callable_):
        self.parent = parent
        self.callable_ = callable_
        self.output_encoding = parent.output_encoding
        self.module = parent.module
        self.encoding_errors = parent.encoding_errors
        self.format_exceptions = parent.format_exceptions
        self.error_handler = parent.error_handler
        self.include_error_handler = parent.include_error_handler
        self.enable_loop = parent.enable_loop
        self.lookup = parent.lookup

    def get_def(self, name):
        return self.parent.get_def(name)