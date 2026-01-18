import numpy
from rdkit import RDRandom as random
from rdkit.ML.Data import Quantize
from rdkit.ML.DecTree import ID3, QuantTree
from rdkit.ML.InfoTheory import entropy
def TestTree():
    """ testing code for named trees

    """
    examples1 = [['p1', 0, 1, 0, 0], ['p2', 0, 0, 0, 1], ['p3', 0, 0, 1, 2], ['p4', 0, 1, 1, 2], ['p5', 1, 0, 0, 2], ['p6', 1, 0, 1, 2], ['p7', 1, 1, 0, 2], ['p8', 1, 1, 1, 0]]
    attrs = list(range(1, len(examples1[0]) - 1))
    nPossibleVals = [0, 2, 2, 2, 3]
    t1 = ID3.ID3Boot(examples1, attrs, nPossibleVals, maxDepth=1)
    t1.Print()