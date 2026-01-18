from sys import version_info as _swig_python_version_info
import weakref
def WriteAssignment(self, file_name):
    """
        Writes the current solution to a file containing an AssignmentProto.
        Returns false if the file cannot be opened or if there is no current
        solution.
        """
    return _pywrapcp.RoutingModel_WriteAssignment(self, file_name)