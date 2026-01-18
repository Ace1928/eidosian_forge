from sys import version_info as _swig_python_version_info
import weakref
def VehicleRouteConsideredVar(self, vehicle):
    """
        Returns the variable specifying whether or not the given vehicle route is
        considered for costs and constraints. It will be equal to 1 iff the route
        of the vehicle is not empty OR vehicle_used_when_empty_[vehicle] is true.
        """
    return _pywrapcp.RoutingModel_VehicleRouteConsideredVar(self, vehicle)