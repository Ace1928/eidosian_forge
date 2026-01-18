from sys import version_info as _swig_python_version_info
import weakref
def AddVariableTargetToFinalizer(self, var, target):
    """
        Add a variable to set the closest possible to the target value in the
        solution finalizer.
        """
    return _pywrapcp.RoutingModel_AddVariableTargetToFinalizer(self, var, target)