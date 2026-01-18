from sys import version_info as _swig_python_version_info
import weakref
def IndexOfConstraint(self, vars, index, target):
    """
        This constraint is a special case of the element constraint with
        an array of integer variables, where the variables are all
        different and the index variable is constrained such that
        vars[index] == target.
        """
    return _pywrapcp.Solver_IndexOfConstraint(self, vars, index, target)