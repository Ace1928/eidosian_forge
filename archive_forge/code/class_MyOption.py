from CreateFps import GetMolFingerprint
from rdkit import Chem, DataStructs
from rdkit.ML.KNN.KNNRegressionModel import KNNRegressionModel
from rdkit.RDLogger import logger
import sys
import copy
import types
from optparse import Option, OptionParser, OptionValueError
class MyOption(Option):
    TYPES = Option.TYPES + ('floatlist',)
    TYPE_CHECKER = copy.copy(Option.TYPE_CHECKER)
    TYPE_CHECKER['floatlist'] = check_floatlist