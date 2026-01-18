from sys import version_info as _swig_python_version_info
import weakref
def GetPerfectBinaryDisjunctions(self):
    """
        Returns the list of all perfect binary disjunctions, as pairs of variable
        indices: a disjunction is "perfect" when its variables do not appear in
        any other disjunction. Each pair is sorted (lowest variable index first),
        and the output vector is also sorted (lowest pairs first).
        """
    return _pywrapcp.RoutingModel_GetPerfectBinaryDisjunctions(self)