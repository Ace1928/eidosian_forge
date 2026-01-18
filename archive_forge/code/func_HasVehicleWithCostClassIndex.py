from sys import version_info as _swig_python_version_info
import weakref
def HasVehicleWithCostClassIndex(self, cost_class_index):
    """
        Returns true iff the model contains a vehicle with the given
        cost_class_index.
        """
    return _pywrapcp.RoutingModel_HasVehicleWithCostClassIndex(self, cost_class_index)