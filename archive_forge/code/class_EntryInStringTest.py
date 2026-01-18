from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class EntryInStringTest(ParserTest, TestCase):
    input_string = u'\n        @article{Me2010, author="Brett, Matthew", title="An article\n        @article{something, author={Name, Another}, title={not really an article}}\n        "}\n        @article{Me2009,author={Nom de Plume, My}, title="A short story"}\n    '
    correct_result = BibliographyData(entries=[(u'Me2010', Entry(u'article', fields=[(u'title', u'An article @article{something, author={Name, Another}, title={not really an article}}')], persons=[(u'author', [Person(u'Brett, Matthew')])])), (u'Me2009', Entry(u'article', fields=[(u'title', u'A short story')], persons={u'author': [Person(u'Nom de Plume, My')]}))])