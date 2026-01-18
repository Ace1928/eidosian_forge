import math
from collections import namedtuple
def CalcEnrichment(scores, col, fractions):
    """ Determines the enrichment factor for a set of fractions """
    numMol = len(scores)
    if numMol == 0:
        raise ValueError('score list is empty')
    if len(fractions) == 0:
        raise ValueError('fraction list is empty')
    for i in fractions:
        if i > 1 or i < 0:
            raise ValueError('fractions must be between [0,1]')
    numPerFrac = [math.ceil(numMol * f) for f in fractions]
    numPerFrac.append(numMol)
    numActives = 0
    enrich = []
    for i in range(numMol):
        if i > numPerFrac[0] - 1 and i > 0:
            enrich.append(1.0 * numActives * numMol / i)
            numPerFrac.pop(0)
        active = scores[i][col]
        if active:
            numActives += 1
    if numActives > 0:
        enrich = [e / numActives for e in enrich]
    else:
        enrich = [0.0] * len(fractions)
    return enrich