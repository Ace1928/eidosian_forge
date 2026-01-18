from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
class NamePart(object):

    def __init__(self, part, abbr=False):
        self.part = part
        self.abbr = abbr

    def __repr__(self):
        abbr = 'abbr' if self.abbr else ''
        return 'person.%s(%s)' % (self.part, abbr)