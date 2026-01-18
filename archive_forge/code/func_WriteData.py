import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def WriteData(outFile, varNames, qBounds, examples):
    """ writes out a .qdat file

      **Arguments**

        - outFile: a file object

        - varNames: a list of variable names

        - qBounds: the list of quantization bounds (should be the same length
           as _varNames_)

        - examples: the data to be written

    """
    outFile.write('# Quantized data from DataUtils\n')
    outFile.write('# ----------\n')
    outFile.write('# Variable Table\n')
    for i in range(len(varNames)):
        outFile.write('# %s %s\n' % (varNames[i], str(qBounds[i])))
    outFile.write('# ----------\n')
    for example in examples:
        outFile.write(' '.join([str(e) for e in example]) + '\n')