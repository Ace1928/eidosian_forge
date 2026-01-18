from sys import version_info as _swig_python_version_info
import weakref
def IgnoreDisjunctionsAlreadyForcedToZero(self):
    """
        SPECIAL: Makes the solver ignore all the disjunctions whose active
        variables are all trivially zero (i.e. Max() == 0), by setting their
        max_cardinality to 0.
        This can be useful when using the BaseBinaryDisjunctionNeighborhood
        operators, in the context of arc-based routing.
        """
    return _pywrapcp.RoutingModel_IgnoreDisjunctionsAlreadyForcedToZero(self)