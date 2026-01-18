from sys import version_info as _swig_python_version_info
import weakref
def ScheduleOrPostpone(self, var, est, marker):
    """
        Returns a decision that tries to schedule a task at a given time.
        On the Apply branch, it will set that interval var as performed and set
        its start to 'est'. On the Refute branch, it will just update the
        'marker' to 'est' + 1. This decision is used in the
        INTERVAL_SET_TIMES_FORWARD strategy.
        """
    return _pywrapcp.Solver_ScheduleOrPostpone(self, var, est, marker)