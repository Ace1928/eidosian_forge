from sys import version_info as _swig_python_version_info
import weakref
def RankFirstInterval(self, sequence, index):
    """
        Returns a decision that tries to rank first the ith interval var
        in the sequence variable.
        """
    return _pywrapcp.Solver_RankFirstInterval(self, sequence, index)