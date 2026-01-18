import math
from collections import namedtuple
def CalcAUC(scores, col):
    """ Determines the area under the ROC curve """
    roc = CalcROC(scores, col)
    FPR = roc.FPR
    TPR = roc.TPR
    numMol = len(scores)
    AUC = 0
    for i in range(0, numMol - 1):
        AUC += (FPR[i + 1] - FPR[i]) * (TPR[i + 1] + TPR[i])
    return 0.5 * AUC