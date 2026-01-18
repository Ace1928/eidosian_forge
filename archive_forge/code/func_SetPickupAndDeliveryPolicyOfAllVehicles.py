from sys import version_info as _swig_python_version_info
import weakref
def SetPickupAndDeliveryPolicyOfAllVehicles(self, policy):
    """
        Sets the Pickup and delivery policy of all vehicles. It is equivalent to
        calling SetPickupAndDeliveryPolicyOfVehicle on all vehicles.
        """
    return _pywrapcp.RoutingModel_SetPickupAndDeliveryPolicyOfAllVehicles(self, policy)