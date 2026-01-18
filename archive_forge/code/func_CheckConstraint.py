from sys import version_info as _swig_python_version_info
import weakref
def CheckConstraint(self, ct):
    """
        Checks whether adding this constraint will lead to an immediate
        failure. It will return false if the model is already inconsistent, or if
        adding the constraint makes it inconsistent.
        """
    return _pywrapcp.Solver_CheckConstraint(self, ct)