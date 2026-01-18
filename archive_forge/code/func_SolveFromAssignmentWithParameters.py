from sys import version_info as _swig_python_version_info
import weakref
def SolveFromAssignmentWithParameters(self, assignment, search_parameters, solutions=None):
    """
        Same as above, except that if assignment is not null, it will be used as
        the initial solution.
        """
    return _pywrapcp.RoutingModel_SolveFromAssignmentWithParameters(self, assignment, search_parameters, solutions)