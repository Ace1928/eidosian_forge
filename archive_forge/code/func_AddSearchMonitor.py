from sys import version_info as _swig_python_version_info
import weakref
def AddSearchMonitor(self, monitor):
    """ Adds a search monitor to the search used to solve the routing model."""
    return _pywrapcp.RoutingModel_AddSearchMonitor(self, monitor)