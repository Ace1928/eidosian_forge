from sys import version_info as _swig_python_version_info
import weakref
def AddWeightedSumEqualVarDimension(self, *args):
    """
        *Overload 1:*
        This dimension imposes that for all bins b, the weighted sum
        (weights[i]) of all objects i assigned to 'b' is equal to loads[b].

        |

        *Overload 2:*
        This dimension imposes that for all bins b, the weighted sum
        (weights->Run(i, b)) of all objects i assigned to 'b' is equal to
        loads[b].
        """
    return _pywrapcp.Pack_AddWeightedSumEqualVarDimension(self, *args)