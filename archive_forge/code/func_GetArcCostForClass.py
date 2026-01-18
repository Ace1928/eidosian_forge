from sys import version_info as _swig_python_version_info
import weakref
def GetArcCostForClass(self, from_index, to_index, cost_class_index):
    """
        Returns the cost of the segment between two nodes for a given cost
        class. Input are variable indices of nodes and the cost class.
        Unlike GetArcCostForVehicle(), if cost_class is kNoCost, then the
        returned cost won't necessarily be zero: only some of the components
        of the cost that depend on the cost class will be omited. See the code
        for details.
        """
    return _pywrapcp.RoutingModel_GetArcCostForClass(self, from_index, to_index, cost_class_index)