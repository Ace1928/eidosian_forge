import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def CombiEnum(sequence):
    """ This generator takes a sequence of sequences as an argument and
  provides all combinations of the elements of the subsequences:

  >>> gen = CombiEnum(((1, 2), (10, 20)))
  >>> next(gen)
  [1, 10]
  >>> next(gen)
  [1, 20]

  >>> [x for x in CombiEnum(((1, 2), (10,20)))]
  [[1, 10], [1, 20], [2, 10], [2, 20]]

  >>> [x for x in CombiEnum(((1, 2),(10, 20), (100, 200)))]
  [[1, 10, 100], [1, 10, 200], [1, 20, 100], [1, 20, 200], [2, 10, 100],
   [2, 10, 200], [2, 20, 100], [2, 20, 200]]

  """
    if not len(sequence):
        yield []
    elif len(sequence) == 1:
        for entry in sequence[0]:
            yield [entry]
    else:
        for entry in sequence[0]:
            for subVal in CombiEnum(sequence[1:]):
                yield ([entry] + subVal)