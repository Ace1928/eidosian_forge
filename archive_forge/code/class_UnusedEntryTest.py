from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class UnusedEntryTest(ParserTest, TestCase):
    parser_options = {'wanted_entries': []}
    input_string = u'\n        @Article(\n            gsl,\n            author = nobody,\n        )\n    '
    correct_result = BibliographyData()