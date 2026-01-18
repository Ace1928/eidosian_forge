import copy
import numpy
from rdkit.ML.DecTree import CrossValidate, DecTree
def _GetLocalError(node):
    nWrong = 0
    for example in node.GetExamples():
        pred = node.ClassifyExample(example, appendExamples=0)
        if pred != example[-1]:
            nWrong += 1
    return nWrong