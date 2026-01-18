from sys import version_info as _swig_python_version_info
import weakref
class NumericalRevInteger(RevInteger):
    """ Subclass of Rev<T> which adds numerical operations."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, val):
        _pywrapcp.NumericalRevInteger_swiginit(self, _pywrapcp.new_NumericalRevInteger(val))

    def Add(self, s, to_add):
        return _pywrapcp.NumericalRevInteger_Add(self, s, to_add)

    def Incr(self, s):
        return _pywrapcp.NumericalRevInteger_Incr(self, s)

    def Decr(self, s):
        return _pywrapcp.NumericalRevInteger_Decr(self, s)
    __swig_destroy__ = _pywrapcp.delete_NumericalRevInteger