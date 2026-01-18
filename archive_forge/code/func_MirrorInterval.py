from sys import version_info as _swig_python_version_info
import weakref
def MirrorInterval(self, interval_var):
    """
        Creates an interval var that is the mirror image of the given one, that
        is, the interval var obtained by reversing the axis.
        """
    return _pywrapcp.Solver_MirrorInterval(self, interval_var)