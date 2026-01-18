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
def _find_field(self, name, bib_data=None):
    """
        Find the field with the given ``name`` according to this rules:

        - If the given field ``name`` in in ``self.fields``, just return
          self.fields[name].

        - Otherwise, if ``name`` is ``"authors"`` or ``"editors"`` (or any other
          person role), return the list of names as a string, separated by
          ``" and "``.

        - Otherwise, if this entry has a ``crossreff`` field, look up for the
          cross-referenced entry and try to find its field with the given
          ``name``.
        """
    try:
        return self.fields[name]
    except KeyError:
        try:
            return self._find_person_field(name)
        except KeyError:
            return self._find_crossref_field(name, bib_data)