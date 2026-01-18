from sys import version_info as _swig_python_version_info
import weakref
def FixedDurationStartSyncedOnStartIntervalVar(self, interval_var, duration, offset):
    """
        Creates an interval var with a fixed duration whose start is
        synchronized with the start of another interval, with a given
        offset. The performed status is also in sync with the performed
        status of the given interval variable.
        """
    return _pywrapcp.Solver_FixedDurationStartSyncedOnStartIntervalVar(self, interval_var, duration, offset)