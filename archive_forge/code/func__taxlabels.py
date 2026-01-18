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
def _taxlabels(self, options):
    """Get taxon labels (PRIVATE).

        As the taxon names are already in the matrix, this is superfluous
        except for transpose matrices, which are currently unsupported anyway.
        Thus, we ignore the taxlabels command to make handling of duplicate
        taxon names easier.
        """