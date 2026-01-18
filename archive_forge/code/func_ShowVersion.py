from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import BuildComposite, CompositeRun, ScreenComposite
from rdkit.ML.Composite import AdjustComposite
from rdkit.ML.Data import DataUtils, SplitData
def ShowVersion(includeArgs=0):
    """ prints the version number

  """
    print('This is GrowComposite.py version %s' % __VERSION_STRING)
    if includeArgs:
        print('command line was:')
        print(' '.join(sys.argv))