import numpy
from rdkit.ML.Data import Quantize
def ClassifyExamples(self, examples, appendExamples=0):
    preds = []
    for eg in examples:
        pred = self.ClassifyExample(eg, appendExamples)
        preds.append(int(pred))
    return preds