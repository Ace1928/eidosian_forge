from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class BracesTest(ParserTest, TestCase):
    input_string = u'@ARTICLE{\n                test,\n                title={Polluted\n                    with {DDT}.\n            },\n    }'
    correct_result = BibliographyData([(u'test', Entry('article', [(u'title', 'Polluted with {DDT}.')]))])