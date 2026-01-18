import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
def _setfieldnames(self, names, titles):
    """convert input field names into a list and assign to the _names
        attribute """
    if names:
        if type(names) in [list, tuple]:
            pass
        elif isinstance(names, str):
            names = names.split(',')
        else:
            raise NameError('illegal input names %s' % repr(names))
        self._names = [n.strip() for n in names[:self._nfields]]
    else:
        self._names = []
    self._names += ['f%d' % i for i in range(len(self._names), self._nfields)]
    _dup = find_duplicate(self._names)
    if _dup:
        raise ValueError('Duplicate field names: %s' % _dup)
    if titles:
        self._titles = [n.strip() for n in titles[:self._nfields]]
    else:
        self._titles = []
        titles = []
    if self._nfields > len(titles):
        self._titles += [None] * (self._nfields - len(titles))