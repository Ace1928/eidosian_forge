import numpy
from rdkit import RDRandom as random
from rdkit.ML.Data import Quantize
from rdkit.ML.DecTree import ID3, QuantTree
from rdkit.ML.InfoTheory import entropy
def TestQuantTree():
    """ Testing code for named trees

    The created pkl file is required by the unit test code.
    """
    examples1 = [['p1', 0, 1, 0.1, 0], ['p2', 0, 0, 0.1, 1], ['p3', 0, 0, 1.1, 2], ['p4', 0, 1, 1.1, 2], ['p5', 1, 0, 0.1, 2], ['p6', 1, 0, 1.1, 2], ['p7', 1, 1, 0.1, 2], ['p8', 1, 1, 1.1, 0]]
    attrs = list(range(1, len(examples1[0]) - 1))
    nPossibleVals = [0, 2, 2, 0, 3]
    boundsPerVar = [0, 0, 0, 1, 0]
    print('base')
    t1 = QuantTreeBoot(examples1, attrs, nPossibleVals, boundsPerVar)
    t1.Pickle('test_data/QuantTree1.pkl')
    t1.Print()
    print('depth limit')
    t1 = QuantTreeBoot(examples1, attrs, nPossibleVals, boundsPerVar, maxDepth=1)
    t1.Pickle('test_data/QuantTree1.pkl')
    t1.Print()