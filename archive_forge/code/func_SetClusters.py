from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers as rdsimdiv
def SetClusters(self, clusters):
    assert len(clusters) == self._nClusters
    self._clusters = clusters