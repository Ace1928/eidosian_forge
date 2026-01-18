from sys import version_info as _swig_python_version_info
import weakref
def BestValueSolutionCollector(self, *args):
    """
        *Overload 1:*
        Collect the solution corresponding to the optimal value of the objective
        of 'assignment'; if 'assignment' does not have an objective no solution is
        collected. This collector only collects one solution corresponding to the
        best objective value (the first one found).

        |

        *Overload 2:*
        Collect the solution corresponding to the optimal value of the
        objective of the internal assignment; if this assignment does not have an
        objective no solution is collected. This collector only collects one
        solution corresponding to the best objective value (the first one found).
        The variables and objective(s) will need to be added later.
        """
    return _pywrapcp.Solver_BestValueSolutionCollector(self, *args)