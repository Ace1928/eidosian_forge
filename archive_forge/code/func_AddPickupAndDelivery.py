from sys import version_info as _swig_python_version_info
import weakref
def AddPickupAndDelivery(self, pickup, delivery):
    """
        Notifies that index1 and index2 form a pair of nodes which should belong
        to the same route. This methods helps the search find better solutions,
        especially in the local search phase.
        It should be called each time you have an equality constraint linking
        the vehicle variables of two node (including for instance pickup and
        delivery problems):
            Solver* const solver = routing.solver();
            int64_t index1 = manager.NodeToIndex(node1);
            int64_t index2 = manager.NodeToIndex(node2);
            solver->AddConstraint(solver->MakeEquality(
                routing.VehicleVar(index1),
                routing.VehicleVar(index2)));
            routing.AddPickupAndDelivery(index1, index2);
        """
    return _pywrapcp.RoutingModel_AddPickupAndDelivery(self, pickup, delivery)