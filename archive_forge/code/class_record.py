import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
class record(nt.void):
    """A data-type scalar that allows field access as attribute lookup.
    """
    __name__ = 'record'
    __module__ = 'numpy'

    def __repr__(self):
        if _get_legacy_print_mode() <= 113:
            return self.__str__()
        return super().__repr__()

    def __str__(self):
        if _get_legacy_print_mode() <= 113:
            return str(self.item())
        return super().__str__()

    def __getattribute__(self, attr):
        if attr in ('setfield', 'getfield', 'dtype'):
            return nt.void.__getattribute__(self, attr)
        try:
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            obj = self.getfield(*res[:2])
            try:
                dt = obj.dtype
            except AttributeError:
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        else:
            raise AttributeError("'record' object has no attribute '%s'" % attr)

    def __setattr__(self, attr, val):
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError("Cannot set '%s' attribute" % attr)
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            return self.setfield(val, *res[:2])
        elif getattr(self, attr, None):
            return nt.void.__setattr__(self, attr, val)
        else:
            raise AttributeError("'record' object has no attribute '%s'" % attr)

    def __getitem__(self, indx):
        obj = nt.void.__getitem__(self, indx)
        if isinstance(obj, nt.void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        else:
            return obj

    def pprint(self):
        """Pretty-print all fields."""
        names = self.dtype.names
        maxlen = max((len(name) for name in names))
        fmt = '%% %ds: %%s' % maxlen
        rows = [fmt % (name, getattr(self, name)) for name in names]
        return '\n'.join(rows)