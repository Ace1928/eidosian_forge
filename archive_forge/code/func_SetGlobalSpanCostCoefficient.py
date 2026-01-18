from sys import version_info as _swig_python_version_info
import weakref
def SetGlobalSpanCostCoefficient(self, coefficient):
    """
        Sets a cost proportional to the *global* dimension span, that is the
        difference between the largest value of route end cumul variables and
        the smallest value of route start cumul variables.
        In other words:
        global_span_cost =
          coefficient * (Max(dimension end value) - Min(dimension start value)).
        """
    return _pywrapcp.RoutingDimension_SetGlobalSpanCostCoefficient(self, coefficient)