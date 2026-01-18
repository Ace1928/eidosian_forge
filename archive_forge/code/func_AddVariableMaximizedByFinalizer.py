from sys import version_info as _swig_python_version_info
import weakref
def AddVariableMaximizedByFinalizer(self, var):
    """
        Adds a variable to maximize in the solution finalizer (see above for
        information on the solution finalizer).
        """
    return _pywrapcp.RoutingModel_AddVariableMaximizedByFinalizer(self, var)