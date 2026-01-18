from sys import version_info as _swig_python_version_info
import weakref
class RevBool(object):
    """
    This class adds reversibility to a POD type.
    It contains the stamp optimization. i.e. the SaveValue call is done
    only once per node of the search tree.  Please note that actual
    stamps always starts at 1, thus an initial value of 0 will always
    trigger the first SaveValue.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, val):
        _pywrapcp.RevBool_swiginit(self, _pywrapcp.new_RevBool(val))

    def Value(self):
        return _pywrapcp.RevBool_Value(self)

    def SetValue(self, s, val):
        return _pywrapcp.RevBool_SetValue(self, s, val)
    __swig_destroy__ = _pywrapcp.delete_RevBool