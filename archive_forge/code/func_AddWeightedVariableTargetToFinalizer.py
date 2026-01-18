from sys import version_info as _swig_python_version_info
import weakref
def AddWeightedVariableTargetToFinalizer(self, var, target, cost):
    """
        Same as above with a weighted priority: the higher the cost, the more
        priority it has to be set close to the target value.
        """
    return _pywrapcp.RoutingModel_AddWeightedVariableTargetToFinalizer(self, var, target, cost)