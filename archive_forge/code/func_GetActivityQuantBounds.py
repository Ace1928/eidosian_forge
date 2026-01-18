import pickle
import numpy
from rdkit.ML.Data import DataUtils
def GetActivityQuantBounds(self):
    if not hasattr(self, 'activityQuant'):
        self.activityQuant = []
    return self.activityQuant