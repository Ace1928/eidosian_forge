from sys import version_info as _swig_python_version_info
import weakref
def SolutionsLimit(self, solutions):
    """
        Creates a search limit that constrains the number of solutions found
        during the search.
        """
    return _pywrapcp.Solver_SolutionsLimit(self, solutions)