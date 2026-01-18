from sys import version_info as _swig_python_version_info
import weakref
def SetSpanUpperBoundForVehicle(self, upper_bound, vehicle):
    """
        Sets an upper bound on the dimension span on a given vehicle. This is the
        preferred way to limit the "length" of the route of a vehicle according to
        a dimension.
        """
    return _pywrapcp.RoutingDimension_SetSpanUpperBoundForVehicle(self, upper_bound, vehicle)