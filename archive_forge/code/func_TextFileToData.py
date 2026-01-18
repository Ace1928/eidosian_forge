import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def TextFileToData(fName, onlyCols=None):
    """
    #DOC

    """
    ext = fName.split('.')[-1]
    with open(fName, 'r') as inF:
        if ext.upper() == 'CSV':
            splitter = csv.reader(inF)
        else:
            splitter = csv.reader(inF, delimiter='\t')
        res = TextToData(splitter, onlyCols=onlyCols)
    return res