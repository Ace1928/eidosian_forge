from sys import version_info as _swig_python_version_info
import weakref
def SetBreakDistanceDurationOfVehicle(self, distance, duration, vehicle):
    """
        With breaks supposed to be consecutive, this forces the distance between
        breaks of size at least minimum_break_duration to be at most distance.
        This supposes that the time until route start and after route end are
        infinite breaks.
        """
    return _pywrapcp.RoutingDimension_SetBreakDistanceDurationOfVehicle(self, distance, duration, vehicle)