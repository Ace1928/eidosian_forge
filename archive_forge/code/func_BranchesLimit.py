from sys import version_info as _swig_python_version_info
import weakref
def BranchesLimit(self, branches):
    """
        Creates a search limit that constrains the number of branches
        explored in the search tree.
        """
    return _pywrapcp.Solver_BranchesLimit(self, branches)