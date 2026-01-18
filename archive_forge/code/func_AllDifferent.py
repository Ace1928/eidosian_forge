from sys import version_info as _swig_python_version_info
import weakref
def AllDifferent(self, *args):
    """
        *Overload 1:*
        All variables are pairwise different. This corresponds to the
        stronger version of the propagation algorithm.

        |

        *Overload 2:*
        All variables are pairwise different.  If 'stronger_propagation'
        is true, stronger, and potentially slower propagation will
        occur. This API will be deprecated in the future.
        """
    return _pywrapcp.Solver_AllDifferent(self, *args)