from io import StringIO, BytesIO, TextIOWrapper
from collections.abc import Mapping
import sys
import os
import urllib.parse
from email.parser import FeedParser
from email.message import Message
import html
import locale
import tempfile
import warnings
class MiniFieldStorage:
    """Like FieldStorage, for use when no file uploads are possible."""
    filename = None
    list = None
    type = None
    file = None
    type_options = {}
    disposition = None
    disposition_options = {}
    headers = {}

    def __init__(self, name, value):
        """Constructor from field name and value."""
        self.name = name
        self.value = value

    def __repr__(self):
        """Return printable representation."""
        return 'MiniFieldStorage(%r, %r)' % (self.name, self.value)