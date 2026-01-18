from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class CrossFileMacrosTest(ParserTest, TestCase):
    input_strings = [u'@string{jackie = "Jackie Chan"}', u',\n            @Book{\n                i_am_jackie,\n                author = jackie,\n                title = "I Am " # jackie # ": My Life in Action",\n            }\n        ']
    correct_result = BibliographyData({'i_am_jackie': Entry('book', fields=[('title', 'I Am Jackie Chan: My Life in Action')], persons={'author': [Person(u'Chan, Jackie')]})})