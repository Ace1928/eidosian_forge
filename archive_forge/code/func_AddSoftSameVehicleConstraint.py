from sys import version_info as _swig_python_version_info
import weakref
def AddSoftSameVehicleConstraint(self, indices, cost):
    """
        Adds a soft constraint to force a set of variable indices to be on the
        same vehicle. If all nodes are not on the same vehicle, each extra vehicle
        used adds 'cost' to the cost function.
        """
    return _pywrapcp.RoutingModel_AddSoftSameVehicleConstraint(self, indices, cost)