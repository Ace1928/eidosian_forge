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
def _rgroup_sort(r):
    """Sort groups like R1 R2 R10 not R1 R10 R2
    """
    if r[0] == 'R':
        return ('R', int(r[1:]))
    return (r, None)