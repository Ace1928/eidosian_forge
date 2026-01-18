from sys import version_info as _swig_python_version_info
import weakref
def StartExpr(self):
    """
        These methods create expressions encapsulating the start, end
        and duration of the interval var. Please note that these must not
        be used if the interval var is unperformed.
        """
    return _pywrapcp.IntervalVar_StartExpr(self)