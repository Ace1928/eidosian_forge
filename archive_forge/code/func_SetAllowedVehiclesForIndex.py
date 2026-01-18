from sys import version_info as _swig_python_version_info
import weakref
def SetAllowedVehiclesForIndex(self, vehicles, index):
    """
        Sets the vehicles which can visit a given node. If the node is in a
        disjunction, this will not prevent it from being unperformed.
        Specifying an empty vector of vehicles has no effect (all vehicles
        will be allowed to visit the node).
        """
    return _pywrapcp.RoutingModel_SetAllowedVehiclesForIndex(self, vehicles, index)