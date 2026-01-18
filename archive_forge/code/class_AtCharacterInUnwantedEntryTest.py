from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class AtCharacterInUnwantedEntryTest(ParserTest, TestCase):
    parser_options = {'wanted_entries': []}
    input_strings = [',\n            @proceedings{acc,\n                title = {Proc.\\@ of the American Control Conference},\n                notes = "acc@example.org"\n            }\n        ']
    correct_result = BibliographyData()