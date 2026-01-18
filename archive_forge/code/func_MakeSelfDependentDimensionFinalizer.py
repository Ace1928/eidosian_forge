from sys import version_info as _swig_python_version_info
import weakref
def MakeSelfDependentDimensionFinalizer(self, dimension):
    """
        MakeSelfDependentDimensionFinalizer is a finalizer for the slacks of a
        self-dependent dimension. It makes an extensive use of the caches of the
        state dependent transits.
        In detail, MakeSelfDependentDimensionFinalizer returns a composition of a
        local search decision builder with a greedy descent operator for the cumul
        of the start of each route and a guided slack finalizer. Provided there
        are no time windows and the maximum slacks are large enough, once the
        cumul of the start of route is fixed, the guided finalizer can find
        optimal values of the slacks for the rest of the route in time
        proportional to the length of the route. Therefore the composed finalizer
        generally works in time O(log(t)*n*m), where t is the latest possible
        departute time, n is the number of nodes in the network and m is the
        number of vehicles.
        """
    return _pywrapcp.RoutingModel_MakeSelfDependentDimensionFinalizer(self, dimension)