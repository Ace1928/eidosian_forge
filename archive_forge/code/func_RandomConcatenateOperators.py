from sys import version_info as _swig_python_version_info
import weakref
def RandomConcatenateOperators(self, *args):
    """
        *Overload 1:*
        Randomized version of local search concatenator; calls a random operator
        at each call to MakeNextNeighbor().

        |

        *Overload 2:*
        Randomized version of local search concatenator; calls a random operator
        at each call to MakeNextNeighbor(). The provided seed is used to
        initialize the random number generator.
        """
    return _pywrapcp.Solver_RandomConcatenateOperators(self, *args)