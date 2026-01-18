from sys import version_info as _swig_python_version_info
import weakref
def SolveWithIteratedLocalSearch(self, search_parameters):
    """
        Solves the current routing model by using an Iterated Local Search
        approach.
        """
    return _pywrapcp.RoutingModel_SolveWithIteratedLocalSearch(self, search_parameters)