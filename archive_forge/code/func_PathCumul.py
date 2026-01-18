from sys import version_info as _swig_python_version_info
import weakref
def PathCumul(self, *args):
    """
        *Overload 1:*
        Creates a constraint which accumulates values along a path such that:
        cumuls[next[i]] = cumuls[i] + transits[i].
        Active variables indicate if the corresponding next variable is active;
        this could be useful to model unperformed nodes in a routing problem.

        |

        *Overload 2:*
        Creates a constraint which accumulates values along a path such that:
        cumuls[next[i]] = cumuls[i] + transit_evaluator(i, next[i]).
        Active variables indicate if the corresponding next variable is active;
        this could be useful to model unperformed nodes in a routing problem.
        Ownership of transit_evaluator is taken and it must be a repeatable
        callback.

        |

        *Overload 3:*
        Creates a constraint which accumulates values along a path such that:
        cumuls[next[i]] = cumuls[i] + transit_evaluator(i, next[i]) + slacks[i].
        Active variables indicate if the corresponding next variable is active;
        this could be useful to model unperformed nodes in a routing problem.
        Ownership of transit_evaluator is taken and it must be a repeatable
        callback.
        """
    return _pywrapcp.Solver_PathCumul(self, *args)