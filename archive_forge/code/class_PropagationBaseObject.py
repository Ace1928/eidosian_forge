from sys import version_info as _swig_python_version_info
import weakref
class PropagationBaseObject(BaseObject):
    """
    NOLINT
    The PropagationBaseObject is a subclass of BaseObject that is also
    friend to the Solver class. It allows accessing methods useful when
    writing new constraints or new expressions.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, s):
        if self.__class__ == PropagationBaseObject:
            _self = None
        else:
            _self = self
        _pywrapcp.PropagationBaseObject_swiginit(self, _pywrapcp.new_PropagationBaseObject(_self, s))
    __swig_destroy__ = _pywrapcp.delete_PropagationBaseObject

    def DebugString(self):
        return _pywrapcp.PropagationBaseObject_DebugString(self)

    def solver(self):
        return _pywrapcp.PropagationBaseObject_solver(self)

    def Name(self):
        """ Object naming."""
        return _pywrapcp.PropagationBaseObject_Name(self)

    def __disown__(self):
        self.this.disown()
        _pywrapcp.disown_PropagationBaseObject(self)
        return weakref.proxy(self)