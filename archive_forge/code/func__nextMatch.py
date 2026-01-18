from rdkit import DataStructs
from rdkit.DataStructs import TopNContainer
def _nextMatch(self):
    """ *Internal use only* """
    done = 0
    res = None
    sim = 0
    while not done:
        obj = next(self.dataIter)
        fp = self.fingerprinter(obj)
        sim = DataStructs.FingerprintSimilarity(fp, self.probe, self.metric)
        if sim >= self.threshold:
            res = obj
            done = 1
    return (sim, res)