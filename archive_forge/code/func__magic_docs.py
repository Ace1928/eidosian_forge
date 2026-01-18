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
def _magic_docs(self, brief=False, rest=False):
    """Return docstrings from magic functions."""
    mman = self.shell.magics_manager
    docs = mman.lsmagic_docs(brief, missing='No documentation')
    if rest:
        format_string = '**%s%s**::\n\n%s\n\n'
    else:
        format_string = '%s%s:\n%s\n'
    return ''.join([format_string % (magic_escapes['line'], fname, indent(dedent(fndoc))) for fname, fndoc in sorted(docs['line'].items())] + [format_string % (magic_escapes['cell'], fname, indent(dedent(fndoc))) for fname, fndoc in sorted(docs['cell'].items())])