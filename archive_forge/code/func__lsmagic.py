from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
def _lsmagic(self):
    """The main implementation of the %lsmagic"""
    mesc = magic_escapes['line']
    cesc = magic_escapes['cell']
    mman = self.magics_manager
    magics = mman.lsmagic()
    out = ['Available line magics:', mesc + ('  ' + mesc).join(sorted([m for m, v in magics['line'].items() if v not in self.ignore])), '', 'Available cell magics:', cesc + ('  ' + cesc).join(sorted([m for m, v in magics['cell'].items() if v not in self.ignore])), '', mman.auto_status()]
    return '\n'.join(out)