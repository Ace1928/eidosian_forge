import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
def _parseFormats(self, formats, aligned=False):
    """ Parse the field formats """
    if formats is None:
        raise ValueError('Need formats argument')
    if isinstance(formats, list):
        dtype = sb.dtype([('f{}'.format(i), format_) for i, format_ in enumerate(formats)], aligned)
    else:
        dtype = sb.dtype(formats, aligned)
    fields = dtype.fields
    if fields is None:
        dtype = sb.dtype([('f1', dtype)], aligned)
        fields = dtype.fields
    keys = dtype.names
    self._f_formats = [fields[key][0] for key in keys]
    self._offsets = [fields[key][1] for key in keys]
    self._nfields = len(keys)