from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers as rdsimdiv
def MapToClusterScores(self, fp):
    """ Map the fingerprint to a real valued vector of score based on the bit clusters

        The dimension of the vector is same as the number of clusters. Each value in the
        vector corresponds to the number of bits in the corresponding cluster
        that are turned on in the fingerprint

        ARGUMENTS:
         - fp : the fingerprint
        """
    scores = [0] * self._nClusters
    for i, cls in enumerate(self._clusters):
        for bid in cls:
            if fp[bid]:
                scores[i] += 1
    return scores