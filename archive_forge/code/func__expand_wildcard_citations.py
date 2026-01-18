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
@fix_unicode_literals_in_doctest
def _expand_wildcard_citations(self, citations):
    """
        Expand wildcard citations (\\citation{*} in .aux file).

        >>> from pybtex.database import Entry
        >>> data = BibliographyData((
        ...     ('uno', Entry('article')),
        ...     ('dos', Entry('article')),
        ...     ('tres', Entry('article')),
        ...     ('cuatro', Entry('article')),
        ... ))
        >>> list(data._expand_wildcard_citations([]))
        []
        >>> list(data._expand_wildcard_citations(['*']))
        [u'uno', u'dos', u'tres', u'cuatro']
        >>> list(data._expand_wildcard_citations(['uno', '*']))
        [u'uno', u'dos', u'tres', u'cuatro']
        >>> list(data._expand_wildcard_citations(['dos', '*']))
        [u'dos', u'uno', u'tres', u'cuatro']
        >>> list(data._expand_wildcard_citations(['*', 'uno']))
        [u'uno', u'dos', u'tres', u'cuatro']
        >>> list(data._expand_wildcard_citations(['*', 'DOS']))
        [u'uno', u'dos', u'tres', u'cuatro']

        """
    citation_set = CaseInsensitiveSet()
    for citation in citations:
        if citation == '*':
            for key in self.entries:
                if key not in citation_set:
                    citation_set.add(key)
                    yield key
        elif citation not in citation_set:
            citation_set.add(citation)
            yield citation