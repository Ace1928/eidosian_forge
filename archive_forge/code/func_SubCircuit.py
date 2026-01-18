from sys import version_info as _swig_python_version_info
import weakref
def SubCircuit(self, nexts):
    """
        Force the "nexts" variable to create a complete Hamiltonian path
        for those that do not loop upon themselves.
        """
    return _pywrapcp.Solver_SubCircuit(self, nexts)