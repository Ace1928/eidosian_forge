from sys import version_info as _swig_python_version_info
import weakref
class IntervalVarElement(AssignmentElement):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr

    def Var(self):
        return _pywrapcp.IntervalVarElement_Var(self)

    def StartMin(self):
        return _pywrapcp.IntervalVarElement_StartMin(self)

    def StartMax(self):
        return _pywrapcp.IntervalVarElement_StartMax(self)

    def StartValue(self):
        return _pywrapcp.IntervalVarElement_StartValue(self)

    def DurationMin(self):
        return _pywrapcp.IntervalVarElement_DurationMin(self)

    def DurationMax(self):
        return _pywrapcp.IntervalVarElement_DurationMax(self)

    def DurationValue(self):
        return _pywrapcp.IntervalVarElement_DurationValue(self)

    def EndMin(self):
        return _pywrapcp.IntervalVarElement_EndMin(self)

    def EndMax(self):
        return _pywrapcp.IntervalVarElement_EndMax(self)

    def EndValue(self):
        return _pywrapcp.IntervalVarElement_EndValue(self)

    def PerformedMin(self):
        return _pywrapcp.IntervalVarElement_PerformedMin(self)

    def PerformedMax(self):
        return _pywrapcp.IntervalVarElement_PerformedMax(self)

    def PerformedValue(self):
        return _pywrapcp.IntervalVarElement_PerformedValue(self)

    def SetStartMin(self, m):
        return _pywrapcp.IntervalVarElement_SetStartMin(self, m)

    def SetStartMax(self, m):
        return _pywrapcp.IntervalVarElement_SetStartMax(self, m)

    def SetStartRange(self, mi, ma):
        return _pywrapcp.IntervalVarElement_SetStartRange(self, mi, ma)

    def SetStartValue(self, v):
        return _pywrapcp.IntervalVarElement_SetStartValue(self, v)

    def SetDurationMin(self, m):
        return _pywrapcp.IntervalVarElement_SetDurationMin(self, m)

    def SetDurationMax(self, m):
        return _pywrapcp.IntervalVarElement_SetDurationMax(self, m)

    def SetDurationRange(self, mi, ma):
        return _pywrapcp.IntervalVarElement_SetDurationRange(self, mi, ma)

    def SetDurationValue(self, v):
        return _pywrapcp.IntervalVarElement_SetDurationValue(self, v)

    def SetEndMin(self, m):
        return _pywrapcp.IntervalVarElement_SetEndMin(self, m)

    def SetEndMax(self, m):
        return _pywrapcp.IntervalVarElement_SetEndMax(self, m)

    def SetEndRange(self, mi, ma):
        return _pywrapcp.IntervalVarElement_SetEndRange(self, mi, ma)

    def SetEndValue(self, v):
        return _pywrapcp.IntervalVarElement_SetEndValue(self, v)

    def SetPerformedMin(self, m):
        return _pywrapcp.IntervalVarElement_SetPerformedMin(self, m)

    def SetPerformedMax(self, m):
        return _pywrapcp.IntervalVarElement_SetPerformedMax(self, m)

    def SetPerformedRange(self, mi, ma):
        return _pywrapcp.IntervalVarElement_SetPerformedRange(self, mi, ma)

    def SetPerformedValue(self, v):
        return _pywrapcp.IntervalVarElement_SetPerformedValue(self, v)

    def __eq__(self, element):
        return _pywrapcp.IntervalVarElement___eq__(self, element)

    def __ne__(self, element):
        return _pywrapcp.IntervalVarElement___ne__(self, element)
    __swig_destroy__ = _pywrapcp.delete_IntervalVarElement