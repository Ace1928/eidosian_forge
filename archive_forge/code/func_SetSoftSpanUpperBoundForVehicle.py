from sys import version_info as _swig_python_version_info
import weakref
def SetSoftSpanUpperBoundForVehicle(self, bound_cost, vehicle):
    """
        If the span of vehicle on this dimension is larger than bound,
        the cost will be increased by cost * (span - bound).
        """
    return _pywrapcp.RoutingDimension_SetSoftSpanUpperBoundForVehicle(self, bound_cost, vehicle)