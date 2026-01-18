from rdkit.DataStructs.TopNContainer import TopNContainer
def GetNeighbors(self, example):
    """ Returns the k nearest neighbors of the example

    """
    nbrs = TopNContainer(self._k)
    for trex in self._trainingExamples:
        dist = self._dfunc(trex, example, self._attrs)
        if self._radius is None or dist < self._radius:
            nbrs.Insert(-dist, trex)
    nbrs.reverse()
    return [x for x in nbrs]