from sys import version_info as _swig_python_version_info
import weakref
def GetVehicleOfClass(self, vehicle_class):
    """
        Returns a vehicle of the given vehicle class, and -1 if there are no
        vehicles for this class.
        """
    return _pywrapcp.RoutingModel_GetVehicleOfClass(self, vehicle_class)