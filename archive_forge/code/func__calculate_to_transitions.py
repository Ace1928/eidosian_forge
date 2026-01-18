import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
def _calculate_to_transitions(trans_probs):
    """Calculate which 'to transitions' are allowed for each state (PRIVATE).

    This looks through all of the trans_probs, and uses this dictionary
    to determine allowed transitions. It converts this information into
    a dictionary, whose keys are destination states and whose values are
    lists of source states from which the destination is reachable via a
    transition.
    """
    transitions = defaultdict(list)
    for from_state, to_state in trans_probs:
        transitions[to_state].append(from_state)
    return transitions