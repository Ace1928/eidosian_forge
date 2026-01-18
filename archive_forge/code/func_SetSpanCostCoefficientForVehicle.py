from sys import version_info as _swig_python_version_info
import weakref
def SetSpanCostCoefficientForVehicle(self, coefficient, vehicle):
    """
        Sets a cost proportional to the dimension span on a given vehicle,
        or on all vehicles at once. "coefficient" must be nonnegative.
        This is handy to model costs proportional to idle time when the dimension
        represents time.
        The cost for a vehicle is
          span_cost = coefficient * (dimension end value - dimension start value).
        """
    return _pywrapcp.RoutingDimension_SetSpanCostCoefficientForVehicle(self, coefficient, vehicle)