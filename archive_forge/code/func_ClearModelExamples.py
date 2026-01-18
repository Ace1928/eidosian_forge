import pickle
import numpy
from rdkit.ML.Data import DataUtils
def ClearModelExamples(self):
    for i in range(len(self)):
        m = self.GetModel(i)
        try:
            m.ClearExamples()
        except AttributeError:
            pass