from sys import version_info as _swig_python_version_info
import weakref
def GetDepot(self):
    """
        Returns the variable index of the first starting or ending node of all
        routes. If all routes start  and end at the same node (single depot), this
        is the node returned.
        """
    return _pywrapcp.RoutingModel_GetDepot(self)