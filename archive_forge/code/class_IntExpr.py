from sys import version_info as _swig_python_version_info
import weakref
class IntExpr(PropagationBaseObject):
    """
    The class IntExpr is the base of all integer expressions in
    constraint programming.
    It contains the basic protocol for an expression:
      - setting and modifying its bound
      - querying if it is bound
      - listening to events modifying its bounds
      - casting it into a variable (instance of IntVar)
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')

    def Min(self):
        return _pywrapcp.IntExpr_Min(self)

    def SetMin(self, m):
        return _pywrapcp.IntExpr_SetMin(self, m)

    def Max(self):
        return _pywrapcp.IntExpr_Max(self)

    def SetMax(self, m):
        return _pywrapcp.IntExpr_SetMax(self, m)

    def SetRange(self, l, u):
        """ This method sets both the min and the max of the expression."""
        return _pywrapcp.IntExpr_SetRange(self, l, u)

    def SetValue(self, v):
        """ This method sets the value of the expression."""
        return _pywrapcp.IntExpr_SetValue(self, v)

    def Bound(self):
        """ Returns true if the min and the max of the expression are equal."""
        return _pywrapcp.IntExpr_Bound(self)

    def IsVar(self):
        """ Returns true if the expression is indeed a variable."""
        return _pywrapcp.IntExpr_IsVar(self)

    def Var(self):
        """ Creates a variable from the expression."""
        return _pywrapcp.IntExpr_Var(self)

    def VarWithName(self, name):
        """
        Creates a variable from the expression and set the name of the
        resulting var. If the expression is already a variable, then it
        will set the name of the expression, possibly overwriting it.
        This is just a shortcut to Var() followed by set_name().
        """
        return _pywrapcp.IntExpr_VarWithName(self, name)

    def WhenRange(self, *args):
        """
        *Overload 1:*
        Attach a demon that will watch the min or the max of the expression.

        |

        *Overload 2:*
        Attach a demon that will watch the min or the max of the expression.
        """
        return _pywrapcp.IntExpr_WhenRange(self, *args)

    def __repr__(self):
        return _pywrapcp.IntExpr___repr__(self)

    def __str__(self):
        return _pywrapcp.IntExpr___str__(self)

    def __add__(self, *args):
        return _pywrapcp.IntExpr___add__(self, *args)

    def __radd__(self, v):
        return _pywrapcp.IntExpr___radd__(self, v)

    def __sub__(self, *args):
        return _pywrapcp.IntExpr___sub__(self, *args)

    def __rsub__(self, v):
        return _pywrapcp.IntExpr___rsub__(self, v)

    def __mul__(self, *args):
        return _pywrapcp.IntExpr___mul__(self, *args)

    def __rmul__(self, v):
        return _pywrapcp.IntExpr___rmul__(self, v)

    def __floordiv__(self, *args):
        return _pywrapcp.IntExpr___floordiv__(self, *args)

    def __mod__(self, *args):
        return _pywrapcp.IntExpr___mod__(self, *args)

    def __neg__(self):
        return _pywrapcp.IntExpr___neg__(self)

    def __abs__(self):
        return _pywrapcp.IntExpr___abs__(self)

    def Square(self):
        return _pywrapcp.IntExpr_Square(self)

    def __eq__(self, *args):
        return _pywrapcp.IntExpr___eq__(self, *args)

    def __ne__(self, *args):
        return _pywrapcp.IntExpr___ne__(self, *args)

    def __ge__(self, *args):
        return _pywrapcp.IntExpr___ge__(self, *args)

    def __gt__(self, *args):
        return _pywrapcp.IntExpr___gt__(self, *args)

    def __le__(self, *args):
        return _pywrapcp.IntExpr___le__(self, *args)

    def __lt__(self, *args):
        return _pywrapcp.IntExpr___lt__(self, *args)

    def MapTo(self, vars):
        return _pywrapcp.IntExpr_MapTo(self, vars)

    def IndexOf(self, *args):
        return _pywrapcp.IntExpr_IndexOf(self, *args)

    def IsMember(self, values):
        return _pywrapcp.IntExpr_IsMember(self, values)

    def Member(self, values):
        return _pywrapcp.IntExpr_Member(self, values)

    def NotMember(self, starts, ends):
        return _pywrapcp.IntExpr_NotMember(self, starts, ends)