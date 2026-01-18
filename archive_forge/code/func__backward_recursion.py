import warnings
from Bio import BiopythonDeprecationWarning
def _backward_recursion(self, cur_state, sequence_pos, backward_vars):
    """Calculate the value of the backward recursion (PRIVATE).

        Arguments:
         - cur_state -- The letter of the state we are calculating the
           forward variable for.
         - sequence_pos -- The position we are at in the training seq.
         - backward_vars -- The current set of backward variables

        """
    if sequence_pos not in self._s_values:
        self._s_values[sequence_pos] = self._calculate_s_value(sequence_pos, backward_vars)
    state_pos_sum = 0
    have_transition = 0
    for second_state in self._mm.transitions_from(cur_state):
        have_transition = 1
        seq_letter = self._seq.emissions[sequence_pos + 1]
        cur_emission_prob = self._mm.emission_prob[cur_state, seq_letter]
        prev_backward = backward_vars[second_state, sequence_pos + 1]
        cur_transition_prob = self._mm.transition_prob[cur_state, second_state]
        state_pos_sum += cur_emission_prob * prev_backward * cur_transition_prob
    if have_transition:
        return state_pos_sum / self._s_values[sequence_pos]
    else:
        return None