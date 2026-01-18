import warnings
from Bio import BiopythonDeprecationWarning
def _forward_recursion(self, cur_state, sequence_pos, forward_vars):
    """Calculate the value of the forward recursion (PRIVATE).

        Arguments:
         - cur_state -- The letter of the state we are calculating the
           forward variable for.
         - sequence_pos -- The position we are at in the training seq.
         - forward_vars -- The current set of forward variables

        """
    if sequence_pos not in self._s_values:
        self._s_values[sequence_pos] = self._calculate_s_value(sequence_pos, forward_vars)
    seq_letter = self._seq.emissions[sequence_pos]
    cur_emission_prob = self._mm.emission_prob[cur_state, seq_letter]
    scale_emission_prob = cur_emission_prob / self._s_values[sequence_pos]
    state_pos_sum = 0
    have_transition = 0
    for second_state in self._mm.transitions_from(cur_state):
        have_transition = 1
        prev_forward = forward_vars[second_state, sequence_pos - 1]
        cur_trans_prob = self._mm.transition_prob[second_state, cur_state]
        state_pos_sum += prev_forward * cur_trans_prob
    if have_transition:
        return scale_emission_prob * state_pos_sum
    else:
        return None