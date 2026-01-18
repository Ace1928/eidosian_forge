from collections.abc import Mapping, Set
import networkx as nx
class InMultiDegreeView(DiDegreeView):
    """A DegreeView class for inward degree of MultiDiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._pred[n]
        if weight is None:
            return sum((len(data) for data in nbrs.values()))
        return sum((d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()))

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._pred[n]
                deg = sum((len(data) for data in nbrs.values()))
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._pred[n]
                deg = sum((d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()))
                yield (n, deg)