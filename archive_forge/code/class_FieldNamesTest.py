from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class FieldNamesTest(ParserTest, TestCase):
    input_string = u'\n        Check for characters allowed in field names\n        Here the cite key is fine, but the field name is not allowed:\n        ``You are missing a field name``\n        @article{2010, 0author="Me"}\n\n        Underscores allowed (no error)\n        @article{2011, _author="Me"}\n\n        Not so for spaces obviously (``expecting an \'=\'``)\n        @article{2012, author name = "Myself"}\n\n        Or hashes (``missing a field name``)\n        @article{2013, #name = "Myself"}\n\n        But field names can start with +-.\n        @article{2014, .name = "Myself"}\n        @article{2015, +name = "Myself"}\n        @article{2016, -name = "Myself"}\n        @article{2017, @name = "Myself"}\n    '
    correct_result = BibliographyData([('2010', Entry('article')), ('2011', Entry('article', [('_author', 'Me')])), ('2012', Entry('article')), ('2013', Entry('article')), ('2014', Entry('article', [('.name', 'Myself')])), ('2015', Entry('article', [('+name', 'Myself')])), ('2016', Entry('article', [('-name', 'Myself')])), ('2017', Entry('article', [('@name', 'Myself')]))])
    errors = ["syntax error in line 5: '}' expected", "syntax error in line 11: '=' expected", "syntax error in line 14: '}' expected"]