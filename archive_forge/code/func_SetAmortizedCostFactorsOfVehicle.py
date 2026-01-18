from sys import version_info as _swig_python_version_info
import weakref
def SetAmortizedCostFactorsOfVehicle(self, linear_cost_factor, quadratic_cost_factor, vehicle):
    """ Sets the linear and quadratic cost factor of the given vehicle."""
    return _pywrapcp.RoutingModel_SetAmortizedCostFactorsOfVehicle(self, linear_cost_factor, quadratic_cost_factor, vehicle)