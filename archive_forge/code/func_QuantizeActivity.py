import pickle
import numpy
from rdkit.ML.Data import DataUtils
def QuantizeActivity(self, example, activityQuant=None, actCol=-1):
    if activityQuant is None:
        activityQuant = self.activityQuant
    if activityQuant:
        example = example[:]
        act = example[actCol]
        for box in range(len(activityQuant)):
            if act < activityQuant[box]:
                act = box
                break
        else:
            act = box + 1
        example[actCol] = act
    return example