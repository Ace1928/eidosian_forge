import math
import numpy
def GetConfidenceInterval(sd, n, level=95):
    col = tConfs[level]
    dofs = n - 1
    sem = sd / numpy.sqrt(n)
    idx = 0
    while idx < len(tTable) and tTable[idx][0] < dofs:
        idx += 1
    if idx < len(tTable):
        t = tTable[idx][col]
    else:
        t = tTable[-1][col]
    return t * sem