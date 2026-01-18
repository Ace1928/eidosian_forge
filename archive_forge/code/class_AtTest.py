from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class AtTest(ParserTest, TestCase):
    input_string = u'\n        The @ here parses fine in both cases\n        @article{Me2010,\n            title={An @tey article}}\n        @article{Me2009, title="A @tey short story"}\n    '
    correct_result = BibliographyData([('Me2010', Entry('article', [('title', 'An @tey article')])), ('Me2009', Entry('article', [('title', 'A @tey short story')]))])
    errors = ["syntax error in line 2: '(' or '{' expected"]