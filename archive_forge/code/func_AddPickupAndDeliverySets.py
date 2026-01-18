from sys import version_info as _swig_python_version_info
import weakref
def AddPickupAndDeliverySets(self, pickup_disjunction, delivery_disjunction):
    """
        Same as AddPickupAndDelivery but notifying that the performed node from
        the disjunction of index 'pickup_disjunction' is on the same route as the
        performed node from the disjunction of index 'delivery_disjunction'.
        """
    return _pywrapcp.RoutingModel_AddPickupAndDeliverySets(self, pickup_disjunction, delivery_disjunction)