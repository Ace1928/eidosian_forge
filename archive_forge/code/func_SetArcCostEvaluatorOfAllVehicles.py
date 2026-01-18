from sys import version_info as _swig_python_version_info
import weakref
def SetArcCostEvaluatorOfAllVehicles(self, evaluator_index):
    """
        Sets the cost function of the model such that the cost of a segment of a
        route between node 'from' and 'to' is evaluator(from, to), whatever the
        route or vehicle performing the route.
        """
    return _pywrapcp.RoutingModel_SetArcCostEvaluatorOfAllVehicles(self, evaluator_index)