from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
def _computeDeltaWeights(self):
    self.deltaWeights = []
    for i, loc in enumerate(self.locations):
        deltaWeight = {}
        for j, support in enumerate(self.supports[:i]):
            scalar = supportScalar(loc, support)
            if scalar:
                deltaWeight[j] = scalar
        self.deltaWeights.append(deltaWeight)