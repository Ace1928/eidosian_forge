from sys import version_info as _swig_python_version_info
import weakref
class BooleanVar(IntVar):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')
    __repr__ = _swig_repr

    def Min(self):
        return _pywrapcp.BooleanVar_Min(self)

    def SetMin(self, m):
        return _pywrapcp.BooleanVar_SetMin(self, m)

    def Max(self):
        return _pywrapcp.BooleanVar_Max(self)

    def SetMax(self, m):
        return _pywrapcp.BooleanVar_SetMax(self, m)

    def SetRange(self, mi, ma):
        return _pywrapcp.BooleanVar_SetRange(self, mi, ma)

    def Bound(self):
        return _pywrapcp.BooleanVar_Bound(self)

    def Value(self):
        return _pywrapcp.BooleanVar_Value(self)

    def RemoveValue(self, v):
        return _pywrapcp.BooleanVar_RemoveValue(self, v)

    def RemoveInterval(self, l, u):
        return _pywrapcp.BooleanVar_RemoveInterval(self, l, u)

    def WhenBound(self, d):
        return _pywrapcp.BooleanVar_WhenBound(self, d)

    def WhenRange(self, d):
        return _pywrapcp.BooleanVar_WhenRange(self, d)

    def WhenDomain(self, d):
        return _pywrapcp.BooleanVar_WhenDomain(self, d)

    def Size(self):
        return _pywrapcp.BooleanVar_Size(self)

    def Contains(self, v):
        return _pywrapcp.BooleanVar_Contains(self, v)

    def HoleIteratorAux(self, reversible):
        return _pywrapcp.BooleanVar_HoleIteratorAux(self, reversible)

    def DomainIteratorAux(self, reversible):
        return _pywrapcp.BooleanVar_DomainIteratorAux(self, reversible)

    def DebugString(self):
        return _pywrapcp.BooleanVar_DebugString(self)