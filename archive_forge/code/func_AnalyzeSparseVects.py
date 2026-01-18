import numpy
from rdkit.ML.InfoTheory import entropy
def AnalyzeSparseVects(bitVects, actVals):
    """ #DOC

  **Arguments**

    - bitVects: a *sequence* containing SBVs

    - actVals: a *sequence*

   **Returns**

     a list of floats

   **Notes**

      - these need to be bit vects and binary activities

  """
    nPts = len(bitVects)
    if nPts != len(actVals):
        raise ValueError('var and activity lists should be the same length')
    nBits = bitVects[0].GetSize()
    actives = numpy.zeros(nBits, numpy.integer)
    inactives = numpy.zeros(nBits, numpy.integer)
    nActives, nInactives = (0, 0)
    for i in range(nPts):
        sig, act = (bitVects[i], actVals[i])
        onBitList = sig.GetOnBits()
        if act:
            for bit in onBitList:
                actives[bit] += 1
            nActives += 1
        else:
            for bit in onBitList:
                inactives[bit] += 1
            nInactives += 1
    resTbl = numpy.zeros((2, 2), numpy.integer)
    res = []
    gains = []
    for bit in range(nBits):
        nAct, nInact = (actives[bit], inactives[bit])
        if nAct or nInact:
            resTbl[0, 0] = nAct
            resTbl[1, 0] = nPts - nAct
            resTbl[0, 1] = nInact
            resTbl[1, 1] = nPts - nInact
            gain = entropy.InfoGain(resTbl)
            gains.append(gain)
            res.append((bit, gain, nAct, nInact))
    return (res, gains)