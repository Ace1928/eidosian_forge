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
def _seqmatrix2strmatrix(matrix):
    """Convert a Seq-object matrix to a plain sequence-string matrix (PRIVATE)."""
    return {t: str(matrix[t]) for t in matrix}