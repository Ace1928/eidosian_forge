import math
from collections import namedtuple
def CalcROC(scores, col):
    """ Determines a ROC curve """
    numMol = len(scores)
    if numMol == 0:
        raise ValueError('score list is empty')
    TPR = [0] * numMol
    FPR = [0] * numMol
    numActives = 0
    numInactives = 0
    for i in range(numMol):
        if scores[i][col]:
            numActives += 1
        else:
            numInactives += 1
        TPR[i] = numActives
        FPR[i] = numInactives
    if numActives > 0:
        TPR = [1.0 * i / numActives for i in TPR]
    if numInactives > 0:
        FPR = [1.0 * i / numInactives for i in FPR]
    RocCurve = namedtuple('RocCurve', ['FPR', 'TPR'])
    return RocCurve(FPR=FPR, TPR=TPR)