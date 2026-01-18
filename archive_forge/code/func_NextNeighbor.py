from sys import version_info as _swig_python_version_info
import weakref
def NextNeighbor(self, delta, deltadelta):
    """
        OnStart() should really be protected, but then SWIG doesn't see it. So we
        make it public, but only subclasses should access to it (to override it).
        Redefines MakeNextNeighbor to export a simpler interface. The calls to
        ApplyChanges() and RevertChanges() are factored in this method, hiding
        both delta and deltadelta from subclasses which only need to override
        MakeOneNeighbor().
        Therefore this method should not be overridden. Override MakeOneNeighbor()
        instead.
        """
    return _pywrapcp.IntVarLocalSearchOperator_NextNeighbor(self, delta, deltadelta)