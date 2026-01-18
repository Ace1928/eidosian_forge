from __future__ import unicode_literals
import re
from string import ascii_letters, digits
import six
from pybtex import textutils
from pybtex.bibtex.utils import split_name_list
from pybtex.database import Entry, Person, BibliographyDataError
from pybtex.database.input import BaseParser
from pybtex.scanner import (
from pybtex.utils import CaseInsensitiveDict, CaseInsensitiveSet
class DuplicateField(BibliographyDataError):

    def __init__(self, entry_key, field_name):
        message = 'entry with key {} has a duplicate {} field'.format(entry_key, field_name)
        super(DuplicateField, self).__init__(message)