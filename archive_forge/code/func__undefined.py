from __future__ import print_function, unicode_literals
from collections import defaultdict
import six
from pybtex.bibtex.builtins import builtins, print_warning
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import wrap
from pybtex.utils import CaseInsensitiveDict
def _undefined(self):
    raise NotImplementedError