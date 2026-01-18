from collections.abc import Mapping
class UnionMultiInner(UnionAtlas):
    """A read-only union of two inner dicts of MultiAdjacencies.

    The two input dict-of-dict-of-dicts represent the union of
    `G.succ[node]` and `G.pred[node]` for MultiDiGraphs.
    Return values are UnionAtlas.
    The inner level of dict is read-write. But the outer levels are read-only.

    See Also
    ========
    UnionAtlas: View into dict-of-dict
    UnionAdjacency:  View into dict-of-dict-of-dict
    UnionMultiAdjacency:  View into dict-of-dict-of-dict-of-dict
    """
    __slots__ = ()

    def __getitem__(self, node):
        in_succ = node in self._succ
        in_pred = node in self._pred
        if in_succ:
            if in_pred:
                return UnionAtlas(self._succ[node], self._pred[node])
            return UnionAtlas(self._succ[node], {})
        return UnionAtlas({}, self._pred[node])

    def copy(self):
        nodes = set(self._succ.keys()) | set(self._pred.keys())
        return {n: self[n].copy() for n in nodes}