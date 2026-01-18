import numpy
from rdkit.ML.Data import Quantize
def _computeQuantBounds(self):
    neg = len(self._trainingExamples)
    natr = len(self._attrs)
    allVals = numpy.zeros((neg, natr), 'd')
    res = []
    i = 0
    for eg in self._trainingExamples:
        res.append(eg[-1])
        j = 0
        for ai in self._attrs:
            val = eg[ai]
            allVals[i, j] = val
            j += 1
        i += 1
    i = 0
    for ai in self._attrs:
        nbnds = self._qBounds[ai]
        if nbnds > 0:
            mbnds = []
            mgain = -1.0
            for j in range(1, nbnds + 1):
                bnds, igain = Quantize.FindVarMultQuantBounds(allVals[:, i], j, res, self._nClasses)
                if igain > mgain:
                    mbnds = bnds
                    mgain = igain
            self._QBoundVals[ai] = mbnds
        i += 1