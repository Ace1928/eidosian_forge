from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class WantedEntriesTest(ParserTest, TestCase):
    parser_options = {'wanted_entries': ['GSL']}
    input_string = u'\n        @Article(\n            gsl,\n        )\n    '
    correct_result = BibliographyData(entries={'GSL': Entry('article')})