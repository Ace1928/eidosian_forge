from sys import version_info as _swig_python_version_info
import weakref
def IsPickup(self, node_index):
    """ Returns whether the node is a pickup (resp. delivery)."""
    return _pywrapcp.RoutingModel_IsPickup(self, node_index)