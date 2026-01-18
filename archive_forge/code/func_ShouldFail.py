from sys import version_info as _swig_python_version_info
import weakref
def ShouldFail(self):
    """
        These methods are only useful for the SWIG wrappers, which need a way
        to externally cause the Solver to fail.
        """
    return _pywrapcp.Solver_ShouldFail(self)