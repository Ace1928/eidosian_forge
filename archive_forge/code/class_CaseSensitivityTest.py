from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class CaseSensitivityTest(ParserTest, TestCase):
    input_strings = [',\n            @Article{CamelCase,\n                Title = {To CamelCase or Under score},\n                year = 2009,\n                NOTES = "none"\n            }\n        ']
    correct_result = BibliographyData({'CamelCase': Entry('article', fields=[('Title', 'To CamelCase or Under score'), ('year', '2009'), ('NOTES', 'none')])})