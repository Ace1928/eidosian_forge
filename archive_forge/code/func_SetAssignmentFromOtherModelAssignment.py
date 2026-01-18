from sys import version_info as _swig_python_version_info
import weakref
def SetAssignmentFromOtherModelAssignment(self, target_assignment, source_model, source_assignment):
    """
        Given a "source_model" and its "source_assignment", resets
        "target_assignment" with the IntVar variables (nexts_, and vehicle_vars_
        if costs aren't homogeneous across vehicles) of "this" model, with the
        values set according to those in "other_assignment".
        The objective_element of target_assignment is set to this->cost_.
        """
    return _pywrapcp.RoutingModel_SetAssignmentFromOtherModelAssignment(self, target_assignment, source_model, source_assignment)