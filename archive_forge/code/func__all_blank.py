import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def _all_blank(self, first_alphabet, second_alphabet):
    """Return a dictionary with all counts set to zero (PRIVATE).

        This uses the letters in the first and second alphabet to create
        a dictionary with keys of two tuples organized as
        (letter of first alphabet, letter of second alphabet). The values
        are all set to 0.
        """
    all_blank = {}
    for first_state in first_alphabet:
        for second_state in second_alphabet:
            all_blank[first_state, second_state] = 0
    return all_blank