from sys import version_info as _swig_python_version_info
import weakref
def DecisionBuilderFromAssignment(self, assignment, db, vars):
    """
        Returns a decision builder for which the left-most leaf corresponds
        to assignment, the rest of the tree being explored using 'db'.
        """
    return _pywrapcp.Solver_DecisionBuilderFromAssignment(self, assignment, db, vars)