from sys import version_info as _swig_python_version_info
import weakref
def RoutesToAssignment(self, routes, ignore_inactive_indices, close_routes, assignment):
    """
        Fills an assignment from a specification of the routes of the
        vehicles. The routes are specified as lists of variable indices that
        appear on the routes of the vehicles. The indices of the outer vector in
        'routes' correspond to vehicles IDs, the inner vector contains the
        variable indices on the routes for the given vehicle. The inner vectors
        must not contain the start and end indices, as these are determined by the
        routing model.  Sets the value of NextVars in the assignment, adding the
        variables to the assignment if necessary. The method does not touch other
        variables in the assignment. The method can only be called after the model
        is closed.  With ignore_inactive_indices set to false, this method will
        fail (return nullptr) in case some of the route contain indices that are
        deactivated in the model; when set to true, these indices will be
        skipped.  Returns true if routes were successfully
        loaded. However, such assignment still might not be a valid
        solution to the routing problem due to more complex constraints;
        it is advisible to call solver()->CheckSolution() afterwards.
        """
    return _pywrapcp.RoutingModel_RoutesToAssignment(self, routes, ignore_inactive_indices, close_routes, assignment)