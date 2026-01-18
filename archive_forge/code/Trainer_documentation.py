import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
Add transitions from the training sequence to the current counts (PRIVATE).

        Arguments:
         - state_seq -- A Seq object with the states of the current training
           sequence.
         - transition_counts -- The current transition counts to add to.

        