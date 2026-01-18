from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def _sort_keys_by_values(p):
    """Return a sorted list of keys of p sorted by values of p (PRIVATE)."""
    return sorted((pn for pn in p if p[pn]), key=lambda pn: p[pn])