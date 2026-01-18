from sys import version_info as _swig_python_version_info
import weakref
def Next(self, assignment, index):
    """
        Assignment inspection
        Returns the variable index of the node directly after the node
        corresponding to 'index' in 'assignment'.
        """
    return _pywrapcp.RoutingModel_Next(self, assignment, index)