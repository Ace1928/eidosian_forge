import copy
import random
import numpy
from rdkit.DataStructs.VectCollection import VectCollection
from rdkit.ML import InfoTheory
from rdkit.ML.DecTree import SigTree
def SigTreeBuilder(examples, attrs, nPossibleVals, initialVar=None, ensemble=None, randomDescriptors=0, **kwargs):
    nRes = nPossibleVals[-1]
    return BuildSigTree(examples, nRes, random=randomDescriptors, **kwargs)