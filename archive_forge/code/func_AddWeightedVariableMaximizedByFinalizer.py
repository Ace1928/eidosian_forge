from sys import version_info as _swig_python_version_info
import weakref
def AddWeightedVariableMaximizedByFinalizer(self, var, cost):
    """
        Adds a variable to maximize in the solution finalizer, with a weighted
        priority: the higher the more priority it has.
        """
    return _pywrapcp.RoutingModel_AddWeightedVariableMaximizedByFinalizer(self, var, cost)