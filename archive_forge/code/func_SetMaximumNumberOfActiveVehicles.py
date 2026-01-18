from sys import version_info as _swig_python_version_info
import weakref
def SetMaximumNumberOfActiveVehicles(self, max_active_vehicles):
    """
        Constrains the maximum number of active vehicles, aka the number of
        vehicles which do not have an empty route. For instance, this can be used
        to limit the number of routes in the case where there are fewer drivers
        than vehicles and that the fleet of vehicle is heterogeneous.
        """
    return _pywrapcp.RoutingModel_SetMaximumNumberOfActiveVehicles(self, max_active_vehicles)