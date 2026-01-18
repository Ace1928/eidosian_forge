import csv
import pickle
import re
import numpy
from rdkit import RDRandom as random
from rdkit.DataStructs import BitUtils
from rdkit.ML.Data import MLData
from rdkit.utils import fileutils
 randomizes the activity values of a dataset

      **Arguments**

        - dataSet: a _ML.Data.MLQuantDataSet_, the activities here will be randomized

        - shuffle: an optional toggle. If this is set, the activity values
          will be shuffled (so the number in each class remains constant)

        - runDetails: an optional CompositeRun object

      **Note**

        - _examples_ are randomized in place


    