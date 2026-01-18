from sys import version_info as _swig_python_version_info
import weakref
def OnStart(self):
    """
        Called by Start() after synchronizing the operator with the current
        assignment. Should be overridden instead of Start() to avoid calling
        IntVarLocalSearchOperator::Start explicitly.
        """
    return _pywrapcp.IntVarLocalSearchOperator_OnStart(self)