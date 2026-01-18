import pickle
import numpy
from rdkit.ML.Data import DataUtils
def SetModelFilterData(self, modelFilterFrac=0.0, modelFilterVal=0.0):
    self._modelFilterFrac = modelFilterFrac
    self._modelFilterVal = modelFilterVal