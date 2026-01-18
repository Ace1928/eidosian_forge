from collections.abc import Mapping
class AdjacencyView(AtlasView):
    """An AdjacencyView is a Read-only Map of Maps of Maps.

    It is a View into a dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.

    See Also
    ========
    AtlasView: View into dict-of-dict
    MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
    """
    __slots__ = ()

    def __getitem__(self, name):
        return AtlasView(self._atlas[name])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}