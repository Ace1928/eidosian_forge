from sys import version_info as _swig_python_version_info
import weakref
def VehicleIndex(self, index):
    """
        Returns the vehicle of the given start/end index, and -1 if the given
        index is not a vehicle start/end.
        """
    return _pywrapcp.RoutingModel_VehicleIndex(self, index)