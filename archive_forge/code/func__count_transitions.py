import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
def _count_transitions(self, state_seq, transition_counts):
    """Add transitions from the training sequence to the current counts (PRIVATE).

        Arguments:
         - state_seq -- A Seq object with the states of the current training
           sequence.
         - transition_counts -- The current transition counts to add to.

        """
    for cur_pos in range(len(state_seq) - 1):
        cur_state = state_seq[cur_pos]
        next_state = state_seq[cur_pos + 1]
        try:
            transition_counts[cur_state, next_state] += 1
        except KeyError:
            raise KeyError(f'Unexpected transition ({cur_state}, {next_state})')
    return transition_counts