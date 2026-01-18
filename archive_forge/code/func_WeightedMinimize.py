from sys import version_info as _swig_python_version_info
import weakref
def WeightedMinimize(self, *args):
    """
        *Overload 1:*
        Creates a minimization weighted objective. The actual objective is
        scalar_prod(sub_objectives, weights).

        |

        *Overload 2:*
        Creates a minimization weighted objective. The actual objective is
        scalar_prod(sub_objectives, weights).
        """
    return _pywrapcp.Solver_WeightedMinimize(self, *args)