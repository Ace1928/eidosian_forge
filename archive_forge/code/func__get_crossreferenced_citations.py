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
def _get_crossreferenced_citations(self, citations, min_crossrefs):
    """
        Get cititations not cited explicitly but referenced by other citations.

        >>> from pybtex.database import Entry
        >>> data = BibliographyData({
        ...     'main_article': Entry('article', {'crossref': 'xrefd_arcicle'}),
        ...     'xrefd_arcicle': Entry('article'),
        ... })
        >>> list(data._get_crossreferenced_citations([], min_crossrefs=1))
        []
        >>> list(data._get_crossreferenced_citations(['main_article'], min_crossrefs=1))
        [u'xrefd_arcicle']
        >>> list(data._get_crossreferenced_citations(['Main_article'], min_crossrefs=1))
        [u'xrefd_arcicle']
        >>> list(data._get_crossreferenced_citations(['main_article'], min_crossrefs=2))
        []
        >>> list(data._get_crossreferenced_citations(['xrefd_arcicle'], min_crossrefs=1))
        []

        >>> data2 = BibliographyData(data.entries, wanted_entries=data.entries.keys())
        >>> list(data2._get_crossreferenced_citations([], min_crossrefs=1))
        []
        >>> list(data2._get_crossreferenced_citations(['main_article'], min_crossrefs=1))
        [u'xrefd_arcicle']
        >>> list(data2._get_crossreferenced_citations(['Main_article'], min_crossrefs=1))
        [u'xrefd_arcicle']
        >>> list(data2._get_crossreferenced_citations(['main_article'], min_crossrefs=2))
        []
        >>> list(data2._get_crossreferenced_citations(['xrefd_arcicle'], min_crossrefs=1))
        []
        >>> list(data2._get_crossreferenced_citations(['xrefd_arcicle'], min_crossrefs=1))
        []

        """
    crossref_count = CaseInsensitiveDefaultDict(int)
    citation_set = CaseInsensitiveSet(citations)
    for citation in citations:
        try:
            entry = self.entries[citation]
            crossref = entry.fields['crossref']
        except KeyError:
            continue
        try:
            crossref_entry = self.entries[crossref]
        except KeyError:
            report_error(BibliographyDataError('bad cross-reference: entry "{key}" refers to entry "{crossref}" which does not exist.'.format(key=citation, crossref=crossref)))
            continue
        canonical_crossref = crossref_entry.key
        crossref_count[canonical_crossref] += 1
        if crossref_count[canonical_crossref] >= min_crossrefs and canonical_crossref not in citation_set:
            citation_set.add(canonical_crossref)
            yield canonical_crossref