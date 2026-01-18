from sys import version_info as _swig_python_version_info
import weakref
def GetNumberOfDecisionsInFirstSolution(self, search_parameters):
    """
        Returns statistics on first solution search, number of decisions sent to
        filters, number of decisions rejected by filters.
        """
    return _pywrapcp.RoutingModel_GetNumberOfDecisionsInFirstSolution(self, search_parameters)