from sys import version_info as _swig_python_version_info
import weakref
def SetAmortizedCostFactorsOfAllVehicles(self, linear_cost_factor, quadratic_cost_factor):
    """
        The following methods set the linear and quadratic cost factors of
        vehicles (must be positive values). The default value of these parameters
        is zero for all vehicles.

        When set, the cost_ of the model will contain terms aiming at reducing the
        number of vehicles used in the model, by adding the following to the
        objective for every vehicle v:
        INDICATOR(v used in the model) *
          [linear_cost_factor_of_vehicle_[v]
           - quadratic_cost_factor_of_vehicle_[v]*(square of length of route v)]
        i.e. for every used vehicle, we add the linear factor as fixed cost, and
        subtract the square of the route length multiplied by the quadratic
        factor. This second term aims at making the routes as dense as possible.

        Sets the linear and quadratic cost factor of all vehicles.
        """
    return _pywrapcp.RoutingModel_SetAmortizedCostFactorsOfAllVehicles(self, linear_cost_factor, quadratic_cost_factor)