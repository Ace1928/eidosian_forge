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
def _unknown_nexus_block(self, title, contents):
    block = Block()
    block.commandlines.append(contents)
    block.title = title
    self.unknown_blocks.append(block)