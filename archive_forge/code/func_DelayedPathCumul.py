from sys import version_info as _swig_python_version_info
import weakref
def DelayedPathCumul(self, nexts, active, cumuls, transits):
    """
        Delayed version of the same constraint: propagation on the nexts variables
        is delayed until all constraints have propagated.
        """
    return _pywrapcp.Solver_DelayedPathCumul(self, nexts, active, cumuls, transits)