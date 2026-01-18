from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class CrossrefWantedTest(ParserTest, TestCase):
    """When cross-referencing an explicitly cited, the key from .aux file should be used."""
    parser_options = {'wanted_entries': ['GSL', 'GSL2', 'The_Journal']}
    input_string = u'\n        @Article(gsl, crossref="the_journal")\n        @Article(gsl2, crossref="The_Journal")\n        @Journal{the_journal,}\n    '
    correct_result = BibliographyData(entries=[('GSL', Entry('article', [('crossref', 'the_journal')])), ('GSL2', Entry('article', [('crossref', 'The_Journal')])), ('The_Journal', Entry('journal'))])