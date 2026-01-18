from __future__ import absolute_import, unicode_literals
from unittest import TestCase
import six
from six.moves import zip_longest
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
class DuplicateFieldTest(ParserTest, TestCase):
    input_strings = ['\n            @MASTERSTHESIS{\n                Mastering,\n                year = 1364,\n                title = "Mastering Thesis Writing",\n                school = "Charles University in Prague",\n                TITLE = "No One Reads Master\'s Theses Anyway LOL",\n                TiTlE = "Well seriously, lol.",\n            }\n        ']
    correct_result = BibliographyData({'Mastering': Entry('mastersthesis', fields=[('year', '1364'), ('title', 'Mastering Thesis Writing'), ('school', 'Charles University in Prague')])})
    errors = ['entry with key Mastering has a duplicate TITLE field', 'entry with key Mastering has a duplicate TiTlE field']