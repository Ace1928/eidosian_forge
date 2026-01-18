from sys import version_info as _swig_python_version_info
import weakref
def GetDimensionOrDie(self, dimension_name):
    """ Returns a dimension from its name. Dies if the dimension does not exist."""
    return _pywrapcp.RoutingModel_GetDimensionOrDie(self, dimension_name)