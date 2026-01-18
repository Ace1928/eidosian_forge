from sys import version_info as _swig_python_version_info
import weakref
def ComputeLowerBound(self):
    """
        Computes a lower bound to the routing problem solving a linear assignment
        problem. The routing model must be closed before calling this method.
        Note that problems with node disjunction constraints (including optional
        nodes) and non-homogenous costs are not supported (the method returns 0 in
        these cases).
        """
    return _pywrapcp.RoutingModel_ComputeLowerBound(self)