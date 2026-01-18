import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def SortTrees(self, sortOnError=1):
    """ sorts the list of trees

      **Arguments**

        sortOnError: toggles sorting on the trees' errors rather than their counts

    """
    if sortOnError:
        order = numpy.argsort(self.errList)
    else:
        order = numpy.argsort(self.countList)
    self.treeList = [self.treeList[x] for x in order]
    self.countList = [self.countList[x] for x in order]
    self.errList = [self.errList[x] for x in order]