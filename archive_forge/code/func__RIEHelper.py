import math
from collections import namedtuple
def _RIEHelper(scores, col, alpha):
    numMol = len(scores)
    alpha = float(alpha)
    if numMol == 0:
        raise ValueError('score list is empty')
    if alpha <= 0.0:
        raise ValueError('alpha must be greater than zero')
    denom = 1.0 / numMol * ((1 - math.exp(-alpha)) / (math.exp(alpha / numMol) - 1))
    numActives = 0
    sum_exp = 0
    for i in range(numMol):
        active = scores[i][col]
        if active:
            numActives += 1
            sum_exp += math.exp(-(alpha * (i + 1)) / numMol)
    if numActives > 0:
        RIE = sum_exp / (numActives * denom)
    else:
        RIE = 0.0
    return (RIE, numActives)