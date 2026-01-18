import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
class RSlots(object):
    """ Attributes of an R object as a Python mapping.

    R objects can have attributes (slots) that are identified
    by a string key (a name) and that can have any R object
    as the associated value. This class represents a view
    of those attributes that is a Python mapping.

    The proxy to the underlying "parent" R object is held as a
    weak reference. The attributes are therefore not protected
    from garbage collection unless bound to a Python symbol or
    in an other container.
    """
    __slots__ = ['_robj']

    def __init__(self, robj):
        self._robj = weakref.proxy(robj)

    def __getitem__(self, key: str):
        value = self._robj.do_slot(key)
        return conversion.get_conversion().rpy2py(value)

    def __setitem__(self, key: str, value):
        rpy2_value = conversion.get_conversion().py2rpy(value)
        self._robj.do_slot_assign(key, rpy2_value)

    def __len__(self):
        return len(self._robj.list_attrs())

    def keys(self):
        for k in self._robj.list_attrs():
            yield k
    __iter__ = keys

    def items(self):
        for k in self._robj.list_attrs():
            v = self[k]
            yield (k, v)

    def values(self):
        for k in self._robj.list_attrs():
            v = self[k]
            yield v