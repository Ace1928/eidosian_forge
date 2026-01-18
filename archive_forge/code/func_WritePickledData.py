import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def WritePickledData(outName, data):
    """ writes either a .qdat.pkl or a .dat.pkl file

      **Arguments**

        - outName: the name of the file to be used

        - data: either an _MLData.MLDataSet_ or an _MLData.MLQuantDataSet_

    """
    varNames = data.GetVarNames()
    qBounds = data.GetQuantBounds()
    ptNames = data.GetPtNames()
    examples = data.GetAllData()
    with open(outName, 'wb+') as outFile:
        pickle.dump(varNames, outFile)
        pickle.dump(qBounds, outFile)
        pickle.dump(ptNames, outFile)
        pickle.dump(examples, outFile)