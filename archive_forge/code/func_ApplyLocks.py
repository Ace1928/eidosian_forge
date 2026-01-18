from sys import version_info as _swig_python_version_info
import weakref
def ApplyLocks(self, locks):
    """
        Applies a lock chain to the next search. 'locks' represents an ordered
        vector of nodes representing a partial route which will be fixed during
        the next search; it will constrain next variables such that:
        next[locks[i]] == locks[i+1].

        Returns the next variable at the end of the locked chain; this variable is
        not locked. An assignment containing the locks can be obtained by calling
        PreAssignment().
        """
    return _pywrapcp.RoutingModel_ApplyLocks(self, locks)