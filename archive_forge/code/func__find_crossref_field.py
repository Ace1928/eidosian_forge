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
def _find_crossref_field(self, name, bib_data):
    if bib_data is None or 'crossref' not in self.fields:
        raise KeyError(name)
    referenced_entry = bib_data.entries[self.fields['crossref']]
    return referenced_entry._find_field(name, bib_data)