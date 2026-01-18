import math
from collections import namedtuple
def CalcBEDROC(scores, col, alpha):
    """ BEDROC original defined here:
    Truchon, J. & Bayly, C.I.
    Evaluating Virtual Screening Methods: Good and Bad Metric for the "Early Recognition"
    Problem. J. Chem. Inf. Model. 47, 488-508 (2007).
    ** Arguments**

      - scores: 2d list or numpy array
             0th index representing sample
             scores must be in sorted order with low indexes "better"
             scores[sample_id] = vector of sample data
      -  col: int
             Index of sample data which reflects true label of a sample
             scores[sample_id][col] = True iff that sample is active
      -  alpha: float
             hyper parameter from the initial paper for how much to enrich the top
     **Returns**
       float BedROC score
    """
    RIE, numActives = _RIEHelper(scores, col, alpha)
    if numActives > 0:
        numMol = len(scores)
        ratio = 1.0 * numActives / numMol
        RIEmax = (1 - math.exp(-alpha * ratio)) / (ratio * (1 - math.exp(-alpha)))
        RIEmin = (1 - math.exp(alpha * ratio)) / (ratio * (1 - math.exp(alpha)))
        if RIEmax != RIEmin:
            BEDROC = (RIE - RIEmin) / (RIEmax - RIEmin)
        else:
            BEDROC = 1.0
    else:
        BEDROC = 0.0
    return BEDROC