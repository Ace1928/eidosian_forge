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
class BibliographyData(object):

    def __init__(self, entries=None, preamble=None, wanted_entries=None, min_crossrefs=2):
        """
        A :py:class:`.BibliographyData` object contains a dictionary of bibliography
        entries referenced by their keys.
        Each entry represented by an :py:class:`.Entry` object.

        Additionally, :py:class:`.BibliographyData` may contain a LaTeX
        preamble defined by ``@PREAMBLE`` commands in the BibTeX file.
        """
        self.entries = OrderedCaseInsensitiveDict()
        'A dictionary of bibliography entries referenced by their keys.\n\n        The dictionary is case insensitive:\n\n        >>> bib_data = parse_string("""\n        ...     @ARTICLE{gnats,\n        ...         author = {L[eslie] A. Aamport},\n        ...         title = {The Gnats and Gnus Document Preparation System},\n        ...     }\n        ... """, \'bibtex\')\n        >>> bib_data.entries[\'gnats\'] == bib_data.entries[\'GNATS\']\n        True\n\n        '
        self.crossref_count = CaseInsensitiveDefaultDict(int)
        self.min_crossrefs = min_crossrefs
        self._preamble = []
        if wanted_entries is not None:
            self.wanted_entries = CaseInsensitiveSet(wanted_entries)
            self.citations = CaseInsensitiveSet(wanted_entries)
        else:
            self.wanted_entries = None
            self.citations = CaseInsensitiveSet()
        if entries:
            if isinstance(entries, Mapping):
                entries = entries.items()
            for key, entry in entries:
                self.add_entry(key, entry)
        if preamble:
            self._preamble.extend(preamble)

    def __eq__(self, other):
        if not isinstance(other, BibliographyData):
            return super(BibliographyData, self) == other
        return self.entries == other.entries and self._preamble == other._preamble

    def __repr__(self):
        repr_entry = repr(self.entries)
        keys = self.entries.keys()
        for key in keys:
            ind = repr_entry.index(key) - 2
            repr_entry = repr_entry[:ind] + '\n' + repr_entry[ind:]
        repr_entry = indent(repr_entry, prefix='    ')
        repr_entry = repr_entry[4:]
        return 'BibliographyData(\n  entries={0},\n\n  preamble={1})'.format(repr_entry, repr(self._preamble))

    def add_to_preamble(self, *values):
        self._preamble.extend(values)

    @property
    def preamble(self):
        '''
        LaTeX preamble.

        >>> bib_data = parse_string(r"""
        ...     @PREAMBLE{"\\newcommand{\\noopsort}[1]{}"}
        ... """, 'bibtex')
        >>> print(bib_data.preamble)
        \\newcommand{\\noopsort}[1]{}

        .. versionadded:: 0.19
            Earlier versions used :py:meth:`.get_preamble()`, which is now deprecated.
        '''
        return ''.join(self._preamble)

    @deprecated('0.19', 'use BibliographyData.preamble instead')
    def get_preamble(self):
        """
        .. deprecated:: 0.19
            Use :py:attr:`.preamble` instead.
        """
        return self.preamble

    def want_entry(self, key):
        return self.wanted_entries is None or key in self.wanted_entries or '*' in self.wanted_entries

    def get_canonical_key(self, key):
        if key in self.citations:
            return self.citations.get_canonical_key(key)
        else:
            return key

    def add_entry(self, key, entry):
        if not self.want_entry(key):
            return
        if key in self.entries:
            report_error(BibliographyDataError('repeated bibliograhpy entry: %s' % key))
            return
        entry.key = self.get_canonical_key(key)
        self.entries[entry.key] = entry
        try:
            crossref = entry.fields['crossref']
        except KeyError:
            pass
        else:
            if self.wanted_entries is not None:
                self.wanted_entries.add(crossref)

    def add_entries(self, entries):
        for key, entry in entries:
            self.add_entry(key, entry)

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

    def add_extra_citations(self, citations, min_crossrefs):
        expanded_citations = list(self._expand_wildcard_citations(citations))
        crossrefs = list(self._get_crossreferenced_citations(expanded_citations, min_crossrefs))
        return expanded_citations + crossrefs

    def to_string(self, bib_format, **kwargs):
        """
        Return the data as a unicode string in the given format.

        :param bib_format: Data format ("bibtex", "yaml", etc.).

        .. versionadded:: 0.19
        """
        writer = find_plugin('pybtex.database.output', bib_format)(**kwargs)
        return writer.to_string(self)

    @classmethod
    def from_string(cls, value, bib_format, **kwargs):
        """
        Return the data from a unicode string in the given format.

        :param bib_format: Data format ("bibtex", "yaml", etc.).

        .. versionadded:: 0.22.2
        """
        return parse_string(value, bib_format, **kwargs)

    def to_bytes(self, bib_format, **kwargs):
        """
        Return the data as a byte string in the given format.

        :param bib_format: Data format ("bibtex", "yaml", etc.).

        .. versionadded:: 0.19
        """
        writer = find_plugin('pybtex.database.output', bib_format)(**kwargs)
        return writer.to_bytes(self)

    def to_file(self, file, bib_format=None, **kwargs):
        """
        Save the data to a file.

        :param file: A file name or a file-like object.
        :param bib_format: Data format ("bibtex", "yaml", etc.).
            If not specified, Pybtex will try to guess by the file name.

        .. versionadded:: 0.19
        """
        if isinstance(file, six.string_types):
            filename = file
        else:
            filename = getattr(file, 'name', None)
        writer = find_plugin('pybtex.database.output', bib_format, filename=filename)(**kwargs)
        return writer.write_file(self, file)

    @fix_unicode_literals_in_doctest
    def lower(self):
        u'''
        Return another :py:class:`.BibliographyData` with all identifiers converted to lowercase.

        >>> data = parse_string("""
        ...     @BOOK{Obrazy,
        ...         title = "Obrazy z Rus",
        ...         author = "Karel Havlíček Borovský",
        ...     }
        ...     @BOOK{Elegie,
        ...         title = "Tirolské elegie",
        ...         author = "Karel Havlíček Borovský",
        ...     }
        ... """, 'bibtex')
        >>> data_lower = data.lower()
        >>> list(data_lower.entries.keys())
        [u'obrazy', u'elegie']
        >>> for entry in data_lower.entries.values():
        ...     entry.key
        ...     list(entry.persons.keys())
        ...     list(entry.fields.keys())
        u'obrazy'
        [u'author']
        [u'title']
        u'elegie'
        [u'author']
        [u'title']

        '''
        entries_lower = ((key.lower(), entry.lower()) for key, entry in self.entries.items())
        return type(self)(entries=entries_lower, preamble=self._preamble, wanted_entries=self.wanted_entries, min_crossrefs=self.min_crossrefs)