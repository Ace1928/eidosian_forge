from sys import version_info as _swig_python_version_info
import weakref
def AddHardTypeIncompatibility(self, type1, type2):
    """
        Incompatibilities:
        Two nodes with "hard" incompatible types cannot share the same route at
        all, while with a "temporal" incompatibility they can't be on the same
        route at the same time.
        """
    return _pywrapcp.RoutingModel_AddHardTypeIncompatibility(self, type1, type2)