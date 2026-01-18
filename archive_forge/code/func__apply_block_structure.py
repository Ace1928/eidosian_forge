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
def _apply_block_structure(self, title, lines):
    """Apply Block structure to the NEXUS file (PRIVATE)."""
    block = Block('')
    block.title = title
    for line in lines:
        block.commandlines.append(Commandline(line, title))
    self.structured.append(block)