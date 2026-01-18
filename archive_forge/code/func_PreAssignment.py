from sys import version_info as _swig_python_version_info
import weakref
def PreAssignment(self):
    """
        Returns an assignment used to fix some of the variables of the problem.
        In practice, this assignment locks partial routes of the problem. This
        can be used in the context of locking the parts of the routes which have
        already been driven in online routing problems.
        """
    return _pywrapcp.RoutingModel_PreAssignment(self)