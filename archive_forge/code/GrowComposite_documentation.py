from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import BuildComposite, CompositeRun, ScreenComposite
from rdkit.ML.Composite import AdjustComposite
from rdkit.ML.Data import DataUtils, SplitData
 parses command line arguments and updates _runDetails_

      **Arguments**

        - runDetails:  a _CompositeRun.CompositeRun_ object.

  