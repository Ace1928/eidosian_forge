from sys import version_info as _swig_python_version_info
import weakref
def Deviation(self, vars, deviation_var, total_sum):
    """
        Deviation constraint:
        sum_i |n * vars[i] - total_sum| <= deviation_var and
        sum_i vars[i] == total_sum
        n = #vars
        """
    return _pywrapcp.Solver_Deviation(self, vars, deviation_var, total_sum)