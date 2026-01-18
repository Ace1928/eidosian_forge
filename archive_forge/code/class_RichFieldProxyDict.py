from __future__ import unicode_literals
from __future__ import print_function
import re
import six
import textwrap
from pybtex.exceptions import PybtexError
from pybtex.utils import (
from pybtex.richtext import Text
from pybtex.bibtex.utils import split_tex_string, scan_bibtex_string
from pybtex.errors import report_error
from pybtex.py3compat import fix_unicode_literals_in_doctest, python_2_unicode_compatible
from pybtex.plugin import find_plugin
class RichFieldProxyDict(Mapping):

    def __init__(self, fields):
        self._fields = fields

    def __contains__(self):
        return self._fields.__contains__()

    def __iter__(self):
        return self._fields.__iter__()

    def __len__(self):
        return self._fields.__len__()

    def __getitem__(self, key):
        return Text.from_latex(self._fields[key])