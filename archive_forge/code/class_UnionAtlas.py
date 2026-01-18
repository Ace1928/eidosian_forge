from collections.abc import Mapping
class UnionAtlas(Mapping):
    """A read-only union of two atlases (dict-of-dict).

    The two dict-of-dicts represent the inner dict of
    an Adjacency:  `G.succ[node]` and `G.pred[node]`.
    The inner level of dict of both hold attribute key:value
    pairs and is read-write. But the outer level is read-only.

    See Also
    ========
    UnionAdjacency: View into dict-of-dict-of-dict
    UnionMultiAdjacency: View into dict-of-dict-of-dict-of-dict
    """
    __slots__ = ('_succ', '_pred')

    def __getstate__(self):
        return {'_succ': self._succ, '_pred': self._pred}

    def __setstate__(self, state):
        self._succ = state['_succ']
        self._pred = state['_pred']

    def __init__(self, succ, pred):
        self._succ = succ
        self._pred = pred

    def __len__(self):
        return len(self._succ.keys() | self._pred.keys())

    def __iter__(self):
        return iter(set(self._succ.keys()) | set(self._pred.keys()))

    def __getitem__(self, key):
        try:
            return self._succ[key]
        except KeyError:
            return self._pred[key]

    def copy(self):
        result = {nbr: dd.copy() for nbr, dd in self._succ.items()}
        for nbr, dd in self._pred.items():
            if nbr in result:
                result[nbr].update(dd)
            else:
                result[nbr] = dd.copy()
        return result

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return f'{self.__class__.__name__}({self._succ!r}, {self._pred!r})'