from __future__ import print_function, unicode_literals
from collections import defaultdict
import six
from pybtex.bibtex.builtins import builtins, print_warning
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import wrap
from pybtex.utils import CaseInsensitiveDict
class MissingField(str):

    def __new__(cls, name):
        self = str.__new__(cls)
        self.name = name
        return self

    def __nonzero__(self):
        return False