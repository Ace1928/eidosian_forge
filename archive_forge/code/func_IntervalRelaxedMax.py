from sys import version_info as _swig_python_version_info
import weakref
def IntervalRelaxedMax(self, interval_var):
    """
         Creates and returns an interval variable that wraps around the given one,
         relaxing the max start and end. Relaxing means making unbounded when
         optional. If the variable is non optional, this method returns
         interval_var.

         More precisely, such an interval variable behaves as follows:
        When the underlying must be performed, the returned interval variable
             behaves exactly as the underlying;
        When the underlying may or may not be performed, the returned interval
             variable behaves like the underlying, except that it is unbounded on
             the max side;
        When the underlying cannot be performed, the returned interval variable
             is of duration 0 and must be performed in an interval unbounded on
             both sides.

         This is very useful for implementing propagators that may only modify
         the start min or end min.
        """
    return _pywrapcp.Solver_IntervalRelaxedMax(self, interval_var)