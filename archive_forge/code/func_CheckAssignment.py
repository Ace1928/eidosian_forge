from sys import version_info as _swig_python_version_info
import weakref
def CheckAssignment(self, solution):
    """ Checks whether the given assignment satisfies all relevant constraints."""
    return _pywrapcp.Solver_CheckAssignment(self, solution)