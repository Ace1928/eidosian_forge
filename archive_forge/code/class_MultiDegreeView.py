from collections.abc import Mapping, Set
import networkx as nx
class MultiDegreeView(DiDegreeView):
    """A DegreeView class for undirected multigraphs; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return sum((len(keys) for keys in nbrs.values())) + (n in nbrs and len(nbrs[n]))
        deg = sum((d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()))
        if n in nbrs:
            deg += sum((d.get(weight, 1) for d in nbrs[n].values()))
        return deg

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum((len(keys) for keys in nbrs.values())) + (n in nbrs and len(nbrs[n]))
                yield (n, deg)
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum((d.get(weight, 1) for key_dict in nbrs.values() for d in key_dict.values()))
                if n in nbrs:
                    deg += sum((d.get(weight, 1) for d in nbrs[n].values()))
                yield (n, deg)