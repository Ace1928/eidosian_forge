from sys import version_info as _swig_python_version_info
import weakref
def ConstantRestart(self, frequency):
    """
        This search monitor will restart the search periodically after 'frequency'
        failures.
        """
    return _pywrapcp.Solver_ConstantRestart(self, frequency)