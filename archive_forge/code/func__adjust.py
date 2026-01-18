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
def _adjust(set, x, d, leftgreedy=False):
    """Adjust character sets if gaps are inserted (PRIVATE).

            Takes care of new gaps within a coherent character set.
            """
    set.sort()
    addpos = 0
    for i, c in enumerate(set):
        if c >= x:
            set[i] = c + d
        if c == x:
            if leftgreedy or (i > 0 and set[i - 1] == c - 1):
                addpos = i
    if addpos > 0:
        set[addpos:addpos] = list(range(x, x + d))
    return set