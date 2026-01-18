import csv
import itertools
import logging
import math
import re
import sys
from collections import defaultdict, namedtuple
from typing import Generator, List
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, molzip
from rdkit.Chem import rdRGroupDecomposition as rgd
class FreeWilsonDecomposition:
    """FreeWilson decomposition
      rgroups - dictionary of rgroup to list of RGroups
                 i.e. {'Core': [RGroup(...), ...]
                       'R1': [ RGroup(...), RGroup(...)],
                      }
      rgroup_to_descriptor_idx - one hot encoding of smiles to descriptor index
      fitter - scikit learn compatible fitter used for regression
      N - number of rgroups
      r2 - regression r squared
      descriptors - set of the descriptors for molecules in the training set
                    used to not enumerate existing molecules
      row_decomposition - original rgroup decomposition (With row key 'molecule' is an rdkit molecule)
    """

    def __init__(self, rgroups, rgroup_to_descriptor_idx, fitter, r2, descriptors, row_decomposition, num_training, num_reconstructed):
        self.rgroups = rgroups
        self.rgroup_to_descriptor_idx = rgroup_to_descriptor_idx
        self.fitter = fitter
        self.N = len(rgroup_to_descriptor_idx)
        self.r2 = r2
        self.descriptors = set([tuple(d) for d in descriptors])
        self.row_decomposition = row_decomposition
        self.num_training = num_training
        self.num_reconstructed = num_reconstructed