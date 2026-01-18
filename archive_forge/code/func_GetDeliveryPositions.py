from sys import version_info as _swig_python_version_info
import weakref
def GetDeliveryPositions(self, node_index):
    """ Returns the pickup and delivery positions where the node is a delivery."""
    return _pywrapcp.RoutingModel_GetDeliveryPositions(self, node_index)