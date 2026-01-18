from sys import version_info as _swig_python_version_info
import weakref
def NullIntersectExcept(self, first_vars, second_vars, escape_value):
    """
        Creates a constraint that states that all variables in the first
        vector are different from all variables from the second group,
        unless they are assigned to the escape value. Thus the set of
        values in the first vector minus the escape value does not
        intersect with the set of values in the second vector.
        """
    return _pywrapcp.Solver_NullIntersectExcept(self, first_vars, second_vars, escape_value)