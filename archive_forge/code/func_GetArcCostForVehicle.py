from sys import version_info as _swig_python_version_info
import weakref
def GetArcCostForVehicle(self, from_index, to_index, vehicle):
    """
        Returns the cost of the transit arc between two nodes for a given vehicle.
        Input are variable indices of node. This returns 0 if vehicle < 0.
        """
    return _pywrapcp.RoutingModel_GetArcCostForVehicle(self, from_index, to_index, vehicle)