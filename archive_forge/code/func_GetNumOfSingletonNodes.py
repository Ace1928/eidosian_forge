from sys import version_info as _swig_python_version_info
import weakref
def GetNumOfSingletonNodes(self):
    """
        Returns the number of non-start/end nodes which do not appear in a
        pickup/delivery pair.
        """
    return _pywrapcp.RoutingModel_GetNumOfSingletonNodes(self)