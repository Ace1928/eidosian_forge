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
def _parse_nexus_block(self, title, contents):
    """Parse a known Nexus Block (PRIVATE)."""
    self._apply_block_structure(title, contents)
    block = self.structured[-1]
    for line in block.commandlines:
        try:
            getattr(self, '_' + line.command)(line.options)
        except AttributeError:
            raise NexusError(f'Unknown command: {line.command} ') from None