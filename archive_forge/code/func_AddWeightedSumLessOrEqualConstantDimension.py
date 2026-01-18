from sys import version_info as _swig_python_version_info
import weakref
def AddWeightedSumLessOrEqualConstantDimension(self, *args):
    """
        *Overload 1:*
        Dimensions are additional constraints than can restrict what is
        possible with the pack constraint. It can be used to set capacity
        limits, to count objects per bin, to compute unassigned
        penalties...
        This dimension imposes that for all bins b, the weighted sum
        (weights[i]) of all objects i assigned to 'b' is less or equal
        'bounds[b]'.

        |

        *Overload 2:*
        This dimension imposes that for all bins b, the weighted sum
        (weights->Run(i)) of all objects i assigned to 'b' is less or
        equal to 'bounds[b]'. Ownership of the callback is transferred to
        the pack constraint.

        |

        *Overload 3:*
        This dimension imposes that for all bins b, the weighted sum
        (weights->Run(i, b) of all objects i assigned to 'b' is less or
        equal to 'bounds[b]'. Ownership of the callback is transferred to
        the pack constraint.
        """
    return _pywrapcp.Pack_AddWeightedSumLessOrEqualConstantDimension(self, *args)