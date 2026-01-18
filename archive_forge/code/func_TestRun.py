import numpy
from rdkit.ML.Data import SplitData
from rdkit.ML.DecTree import ID3, randomtest
def TestRun():
    """ testing code """
    examples, attrs, nPossibleVals = randomtest.GenRandomExamples(nExamples=200)
    tree, _ = CrossValidationDriver(examples, attrs, nPossibleVals)
    tree.Pickle('save.pkl')
    import copy
    t2 = copy.deepcopy(tree)
    print('t1 == t2', tree == t2)
    l = [tree]
    print('t2 in [tree]', t2 in l, l.index(t2))