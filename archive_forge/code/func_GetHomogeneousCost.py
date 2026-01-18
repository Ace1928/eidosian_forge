from sys import version_info as _swig_python_version_info
import weakref
def GetHomogeneousCost(self, from_index, to_index):
    """
        Returns the cost of the segment between two nodes supposing all vehicle
        costs are the same (returns the cost for the first vehicle otherwise).
        """
    return _pywrapcp.RoutingModel_GetHomogeneousCost(self, from_index, to_index)