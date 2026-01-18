from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class KeylessEntriesTest(ParserTest, TestCase):
    parser_options = {'keyless_entries': True}
    input_string = u'\n        @BOOK(\n            title="I Am Jackie Chan: My Life in Action",\n            year=1999\n        )\n        @BOOK()\n        @BOOK{}\n\n        @BOOK{\n            title = "Der deutsche Jackie Chan Filmführer",\n        }\n\n    '
    correct_result = BibliographyData([('unnamed-1', Entry('book', [('title', 'I Am Jackie Chan: My Life in Action'), ('year', '1999')])), ('unnamed-2', Entry('book')), ('unnamed-3', Entry('book')), ('unnamed-4', Entry('book', [('title', u'Der deutsche Jackie Chan Filmführer')]))])