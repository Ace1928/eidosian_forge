import numpy
from rdkit.ML.DecTree import DecTree
from rdkit.ML.InfoTheory import entropy
 Bootstrapping code for the ID3 algorithm

    see ID3 for descriptions of the arguments

    If _initialVar_ is not set, the algorithm will automatically
     choose the first variable in the tree (the standard greedy
     approach).  Otherwise, _initialVar_ will be used as the first
     split.

  