import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData
def _maketrans(complement_mapping):
    """Make a python string translation table (PRIVATE).

    Arguments:
     - complement_mapping - a dictionary such as ambiguous_dna_complement
       and ambiguous_rna_complement from Data.IUPACData.

    Returns a translation table (a bytes object of length 256) for use with
    the python string's translate method to use in a (reverse) complement.

    Compatible with lower case and upper case sequences.

    For internal use only.
    """
    keys = ''.join(complement_mapping.keys()).encode('ASCII')
    values = ''.join(complement_mapping.values()).encode('ASCII')
    return bytes.maketrans(keys + keys.lower(), values + values.lower())