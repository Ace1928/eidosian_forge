import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class LocationParserError(ValueError):
    """Could not parse a feature location string."""