from sys import version_info as _swig_python_version_info
import weakref
def MoveTowardTargetOperator(self, *args):
    """
        *Overload 1:*
        Creates a local search operator that tries to move the assignment of some
        variables toward a target. The target is given as an Assignment. This
        operator generates neighbors in which the only difference compared to the
        current state is that one variable that belongs to the target assignment
        is set to its target value.

        |

        *Overload 2:*
        Creates a local search operator that tries to move the assignment of some
        variables toward a target. The target is given either as two vectors: a
        vector of variables and a vector of associated target values. The two
        vectors should be of the same length. This operator generates neighbors in
        which the only difference compared to the current state is that one
        variable that belongs to the given vector is set to its target value.
        """
    return _pywrapcp.Solver_MoveTowardTargetOperator(self, *args)