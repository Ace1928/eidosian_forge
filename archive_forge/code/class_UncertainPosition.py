import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class UncertainPosition(ExactPosition):
    """Specify a specific position which is uncertain.

    This is used in UniProt, e.g. ?222 for uncertain position 222, or in the
    XML format explicitly marked as uncertain. Does not apply to GenBank/EMBL.
    """