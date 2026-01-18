from sys import version_info as _swig_python_version_info
import weakref
def SolveWithParameters(self, search_parameters, solutions=None):
    """
        Solves the current routing model with the given parameters. If 'solutions'
        is specified, it will contain the k best solutions found during the search
        (from worst to best, including the one returned by this method), where k
        corresponds to the 'number_of_solutions_to_collect' in
        'search_parameters'. Note that the Assignment returned by the method and
        the ones in solutions are owned by the underlying solver and should not be
        deleted.
        """
    return _pywrapcp.RoutingModel_SolveWithParameters(self, search_parameters, solutions)