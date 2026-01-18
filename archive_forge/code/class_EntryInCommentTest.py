from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class EntryInCommentTest(ParserTest, TestCase):
    input_string = u'\n        Both the articles register despite the comment block\n        @Comment{\n        @article{Me2010, title="An article"}\n        @article{Me2009, title="A short story"}\n        }\n        These all work OK without errors\n        @Comment{and more stuff}\n\n        Last article to show we can get here\n        @article{Me2011, }\n    '
    correct_result = BibliographyData([('Me2010', Entry('article', fields=[('title', 'An article')])), ('Me2009', Entry('article', fields=[('title', 'A short story')])), ('Me2011', Entry('article'))])