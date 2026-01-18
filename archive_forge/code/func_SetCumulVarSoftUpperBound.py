from sys import version_info as _swig_python_version_info
import weakref
def SetCumulVarSoftUpperBound(self, index, upper_bound, coefficient):
    """
        Sets a soft upper bound to the cumul variable of a given variable index.
        If the value of the cumul variable is greater than the bound, a cost
        proportional to the difference between this value and the bound is added
        to the cost function of the model:
          cumulVar <= upper_bound -> cost = 0
           cumulVar > upper_bound -> cost = coefficient * (cumulVar - upper_bound)
        This is also handy to model tardiness costs when the dimension represents
        time.
        """
    return _pywrapcp.RoutingDimension_SetCumulVarSoftUpperBound(self, index, upper_bound, coefficient)