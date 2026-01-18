import copy
import numpy
from rdkit.ML.DecTree import CrossValidate, DecTree
def _testSpecific():
    from rdkit.ML.DecTree import ID3
    oPts = [[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]]
    tPts = oPts + [[0, 1, 1, 0], [0, 1, 1, 0]]
    tree = ID3.ID3Boot(oPts, attrs=range(3), nPossibleVals=[2] * 4)
    tree.Print()
    err, _ = CrossValidate.CrossValidate(tree, oPts)
    print('original error:', err)
    err, _ = CrossValidate.CrossValidate(tree, tPts)
    print('original holdout error:', err)
    newTree, frac2 = PruneTree(tree, oPts, tPts)
    newTree.Print()
    print('best error of pruned tree:', frac2)
    err, badEx = CrossValidate.CrossValidate(newTree, tPts)
    print('pruned holdout error is:', err)
    print(badEx)