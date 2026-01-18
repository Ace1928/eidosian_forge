from sys import version_info as _swig_python_version_info
import weakref
def AddSumVariableWeightsLessOrEqualConstantDimension(self, usage, capacity):
    """
        This dimension imposes:
        forall b in bins,
           sum (i in items: usage[i] * is_assigned(i, b)) <= capacity[b]
        where is_assigned(i, b) is true if and only if item i is assigned
        to the bin b.

        This can be used to model shapes of items by linking variables of
        the same item on parallel dimensions with an allowed assignment
        constraint.
        """
    return _pywrapcp.Pack_AddSumVariableWeightsLessOrEqualConstantDimension(self, usage, capacity)