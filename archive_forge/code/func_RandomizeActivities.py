import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def RandomizeActivities(dataSet, shuffle=0, runDetails=None):
    """ randomizes the activity values of a dataset

      **Arguments**

        - dataSet: a _ML.Data.MLQuantDataSet_, the activities here will be randomized

        - shuffle: an optional toggle. If this is set, the activity values
          will be shuffled (so the number in each class remains constant)

        - runDetails: an optional CompositeRun object

      **Note**

        - _examples_ are randomized in place


    """
    nPts = dataSet.GetNPts()
    if shuffle:
        if runDetails:
            runDetails.shuffled = 1
        acts = dataSet.GetResults()[:]
        random.shuffle(acts, random=random.random)
    else:
        if runDetails:
            runDetails.randomized = 1
        nPossible = dataSet.GetNPossibleVals()[-1]
        acts = [random.randint(0, nPossible) for _ in len(examples)]
    for i in range(nPts):
        tmp = dataSet[i]
        tmp[-1] = acts[i]
        dataSet[i] = tmp