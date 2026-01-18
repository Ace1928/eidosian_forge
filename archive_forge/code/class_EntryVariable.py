from __future__ import print_function, unicode_literals
from collections import defaultdict
import six
from pybtex.bibtex.builtins import builtins, print_warning
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import wrap
from pybtex.utils import CaseInsensitiveDict
class EntryVariable(Variable):

    def __init__(self, interpreter, name):
        Variable.__init__(self)
        self.interpreter = interpreter
        self.name = name

    def set(self, value):
        if value is not None:
            self.validate(value)
            self.interpreter.current_entry_vars[self.name] = value

    def value(self):
        return self.interpreter.current_entry_vars.get(self.name, self.default)