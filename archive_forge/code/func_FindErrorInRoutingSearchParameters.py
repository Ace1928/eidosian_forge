from sys import version_info as _swig_python_version_info
import weakref
def FindErrorInRoutingSearchParameters(search_parameters):
    """
    Returns an empty std::string if the routing search parameters are valid, and
    a non-empty, human readable error description if they're not.
    """
    return _pywrapcp.FindErrorInRoutingSearchParameters(search_parameters)