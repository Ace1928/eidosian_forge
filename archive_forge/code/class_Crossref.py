from __future__ import print_function, unicode_literals
from collections import defaultdict
import six
from pybtex.bibtex.builtins import builtins, print_warning
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import wrap
from pybtex.utils import CaseInsensitiveDict
class Crossref(Field):

    def __init__(self, interpreter):
        super(Crossref, self).__init__(interpreter, 'crossref')

    def value(self):
        try:
            value = self.interpreter.current_entry.fields[self.name]
            crossref_entry = self.interpreter.bib_data.entries[value]
        except KeyError:
            return MissingField(self.name)
        return crossref_entry.key