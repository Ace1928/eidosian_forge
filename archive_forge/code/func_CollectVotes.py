import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def CollectVotes(self, example):
    """ collects votes across every member of the forest for the given example

      **Returns**

        a list of the results

    """
    nTrees = len(self.treeList)
    votes = [0] * nTrees
    for i in range(nTrees):
        votes[i] = self.treeList[i].ClassifyExample(example)
    return votes