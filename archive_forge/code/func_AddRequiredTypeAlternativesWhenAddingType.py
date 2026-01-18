from sys import version_info as _swig_python_version_info
import weakref
def AddRequiredTypeAlternativesWhenAddingType(self, dependent_type, required_type_alternatives):
    """
        If type_D depends on type_R when adding type_D, any node_D of type_D and
        VisitTypePolicy TYPE_ADDED_TO_VEHICLE or
        TYPE_SIMULTANEOUSLY_ADDED_AND_REMOVED requires at least one type_R on its
        vehicle at the time node_D is visited.
        """
    return _pywrapcp.RoutingModel_AddRequiredTypeAlternativesWhenAddingType(self, dependent_type, required_type_alternatives)