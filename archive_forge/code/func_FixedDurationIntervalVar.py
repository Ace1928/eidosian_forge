from sys import version_info as _swig_python_version_info
import weakref
def FixedDurationIntervalVar(self, *args):
    """
        *Overload 1:*
        Creates an interval var with a fixed duration. The duration must
        be greater than 0. If optional is true, then the interval can be
        performed or unperformed. If optional is false, then the interval
        is always performed.

        |

        *Overload 2:*
        Creates a performed interval var with a fixed duration. The duration must
        be greater than 0.

        |

        *Overload 3:*
        Creates an interval var with a fixed duration, and performed_variable.
        The duration must be greater than 0.
        """
    return _pywrapcp.Solver_FixedDurationIntervalVar(self, *args)