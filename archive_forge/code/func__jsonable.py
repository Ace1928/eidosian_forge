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
def _jsonable(self):
    """turn magics dict into jsonable dict of the same structure

        replaces object instances with their class names as strings
        """
    magic_dict = {}
    mman = self.magics_manager
    magics = mman.lsmagic()
    for key, subdict in magics.items():
        d = {}
        magic_dict[key] = d
        for name, obj in subdict.items():
            try:
                classname = obj.__self__.__class__.__name__
            except AttributeError:
                classname = 'Other'
            d[name] = classname
    return magic_dict