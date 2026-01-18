from sys import version_info as _swig_python_version_info
import weakref
def Stamp(self):
    """
        The stamp indicates how many moves in the search tree we have performed.
        It is useful to detect if we need to update same lazy structures.
        """
    return _pywrapcp.Solver_Stamp(self)