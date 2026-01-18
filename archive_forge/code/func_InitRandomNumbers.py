import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
def InitRandomNumbers(seed):
    """ Seeds the random number generators

      **Arguments**

        - seed: a 2-tuple containing integers to be used as the random number seeds

      **Notes**

        this seeds both the RDRandom generator and the one in the standard
        Python _random_ module

    """
    from rdkit import RDRandom
    RDRandom.seed(seed[0])
    random.seed(seed[0])