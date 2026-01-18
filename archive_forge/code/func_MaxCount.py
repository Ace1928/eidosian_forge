import copy
import numpy
from rdkit.ML.DecTree import CrossValidate, DecTree
def MaxCount(examples):
    """ given a set of examples, returns the most common result code

     **Arguments**

        examples: a list of examples to be counted

     **Returns**

       the most common result code

    """
    resList = [x[-1] for x in examples]
    maxVal = max(resList)
    counts = [None] * (maxVal + 1)
    for i in range(maxVal + 1):
        counts[i] = sum([x == i for x in resList])
    return numpy.argmax(counts)