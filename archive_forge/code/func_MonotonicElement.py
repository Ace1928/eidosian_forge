from sys import version_info as _swig_python_version_info
import weakref
def MonotonicElement(self, values, increasing, index):
    """
        Function based element. The constraint takes ownership of the
        callback.  The callback must be monotonic. It must be able to
        cope with any possible value in the domain of 'index'
        (potentially negative ones too). Furtermore, monotonicity is not
        checked. Thus giving a non-monotonic function, or specifying an
        incorrect increasing parameter will result in undefined behavior.
        """
    return _pywrapcp.Solver_MonotonicElement(self, values, increasing, index)