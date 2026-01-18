import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
Return log transform of the given probability dictionary (PRIVATE).

        When calculating the Viterbi equation, add logs of probabilities rather
        than multiplying probabilities, to avoid underflow errors. This method
        returns a new dictionary with the same keys as the given dictionary
        and log-transformed values.
        