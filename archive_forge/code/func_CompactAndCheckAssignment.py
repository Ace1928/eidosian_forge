from sys import version_info as _swig_python_version_info
import weakref
def CompactAndCheckAssignment(self, assignment):
    """
        Same as CompactAssignment() but also checks the validity of the final
        compact solution; if it is not valid, no attempts to repair it are made
        (instead, the method returns nullptr).
        """
    return _pywrapcp.RoutingModel_CompactAndCheckAssignment(self, assignment)