import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def MakeHistogram(self):
    """ creates a histogram of error/count pairs

    """
    nExamples = len(self.treeList)
    histo = []
    i = 1
    lastErr = self.errList[0]
    countHere = self.countList[0]
    eps = 0.001
    while i < nExamples:
        if self.errList[i] - lastErr > eps:
            histo.append((lastErr, countHere))
            lastErr = self.errList[i]
            countHere = self.countList[i]
        else:
            countHere = countHere + self.countList[i]
        i = i + 1
    return histo