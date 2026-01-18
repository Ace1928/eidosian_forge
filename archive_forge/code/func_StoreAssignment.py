from sys import version_info as _swig_python_version_info
import weakref
def StoreAssignment(self, assignment):
    """
        Returns a DecisionBuilder which stores an Assignment
        (calls void Assignment::Store())
        """
    return _pywrapcp.Solver_StoreAssignment(self, assignment)