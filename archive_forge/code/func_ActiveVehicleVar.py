from sys import version_info as _swig_python_version_info
import weakref
def ActiveVehicleVar(self, vehicle):
    """
        Returns the active variable of the vehicle. It will be equal to 1 iff the
        route of the vehicle is not empty, 0 otherwise.
        """
    return _pywrapcp.RoutingModel_ActiveVehicleVar(self, vehicle)