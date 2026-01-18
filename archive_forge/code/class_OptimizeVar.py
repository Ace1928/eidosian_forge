from sys import version_info as _swig_python_version_info
import weakref
class OptimizeVar(object):
    """
    This class encapsulates an objective. It requires the direction
    (minimize or maximize), the variable to optimize, and the
    improvement step.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr

    def Best(self):
        """ Returns the best value found during search."""
        return _pywrapcp.OptimizeVar_Best(self)

    def BeginNextDecision(self, db):
        """ Internal methods."""
        return _pywrapcp.OptimizeVar_BeginNextDecision(self, db)

    def RefuteDecision(self, d):
        return _pywrapcp.OptimizeVar_RefuteDecision(self, d)

    def AtSolution(self):
        return _pywrapcp.OptimizeVar_AtSolution(self)

    def AcceptSolution(self):
        return _pywrapcp.OptimizeVar_AcceptSolution(self)

    def DebugString(self):
        return _pywrapcp.OptimizeVar_DebugString(self)
    __swig_destroy__ = _pywrapcp.delete_OptimizeVar