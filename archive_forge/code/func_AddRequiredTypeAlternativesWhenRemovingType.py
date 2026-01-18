from sys import version_info as _swig_python_version_info
import weakref
def AddRequiredTypeAlternativesWhenRemovingType(self, dependent_type, required_type_alternatives):
    """
        The following requirements apply when visiting dependent nodes that remove
        their type from the route, i.e. type_R must be on the vehicle when type_D
        of VisitTypePolicy ADDED_TYPE_REMOVED_FROM_VEHICLE,
        TYPE_ON_VEHICLE_UP_TO_VISIT or TYPE_SIMULTANEOUSLY_ADDED_AND_REMOVED is
        visited.
        """
    return _pywrapcp.RoutingModel_AddRequiredTypeAlternativesWhenRemovingType(self, dependent_type, required_type_alternatives)