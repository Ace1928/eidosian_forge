import warnings
from Bio import BiopythonDeprecationWarning
def _calculate_s_value(self, seq_pos, previous_vars):
    """Calculate the next scaling variable for a sequence position (PRIVATE).

        This utilizes the approach of choosing s values such that the
        sum of all of the scaled f values is equal to 1.

        Arguments:
         - seq_pos -- The current position we are at in the sequence.
         - previous_vars -- All of the forward or backward variables
           calculated so far.

        Returns:
         - The calculated scaling variable for the sequence item.

        """
    state_letters = self._mm.state_alphabet
    s_value = 0
    for main_state in state_letters:
        emission = self._mm.emission_prob[main_state, self._seq.emissions[seq_pos]]
        trans_and_var_sum = 0
        for second_state in self._mm.transitions_from(main_state):
            var_value = previous_vars[second_state, seq_pos - 1]
            trans_value = self._mm.transition_prob[second_state, main_state]
            trans_and_var_sum += var_value * trans_value
        s_value += emission * trans_and_var_sum
    return s_value