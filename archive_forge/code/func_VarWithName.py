from sys import version_info as _swig_python_version_info
import weakref
def VarWithName(self, name):
    """
        Creates a variable from the expression and set the name of the
        resulting var. If the expression is already a variable, then it
        will set the name of the expression, possibly overwriting it.
        This is just a shortcut to Var() followed by set_name().
        """
    return _pywrapcp.IntExpr_VarWithName(self, name)