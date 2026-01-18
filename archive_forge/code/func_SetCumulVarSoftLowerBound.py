from sys import version_info as _swig_python_version_info
import weakref
def SetCumulVarSoftLowerBound(self, index, lower_bound, coefficient):
    """
        Sets a soft lower bound to the cumul variable of a given variable index.
        If the value of the cumul variable is less than the bound, a cost
        proportional to the difference between this value and the bound is added
        to the cost function of the model:
          cumulVar > lower_bound -> cost = 0
          cumulVar <= lower_bound -> cost = coefficient * (lower_bound -
                      cumulVar).
        This is also handy to model earliness costs when the dimension represents
        time.
        """
    return _pywrapcp.RoutingDimension_SetCumulVarSoftLowerBound(self, index, lower_bound, coefficient)