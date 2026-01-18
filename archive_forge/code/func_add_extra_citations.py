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
def add_extra_citations(self, citations, min_crossrefs):
    expanded_citations = list(self._expand_wildcard_citations(citations))
    crossrefs = list(self._get_crossreferenced_citations(expanded_citations, min_crossrefs))
    return expanded_citations + crossrefs