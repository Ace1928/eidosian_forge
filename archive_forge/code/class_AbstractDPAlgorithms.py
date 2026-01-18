import warnings
from Bio import BiopythonDeprecationWarning
class AbstractDPAlgorithms:
    """An abstract class to calculate forward and backward probabilities.

    This class should not be instantiated directly, but should be used
    through a derived class which implements proper scaling of variables.

    This class is just meant to encapsulate the basic forward and backward
    algorithms, and allow derived classes to deal with the problems of
    multiplying probabilities.

    Derived class of this must implement:

    - _forward_recursion -- Calculate the forward values in the recursion
      using some kind of technique for preventing underflow errors.
    - _backward_recursion -- Calculate the backward values in the recursion
      step using some technique to prevent underflow errors.

    """

    def __init__(self, markov_model, sequence):
        """Initialize to calculate forward and backward probabilities.

        Arguments:
         - markov_model -- The current Markov model we are working with.
         - sequence -- A training sequence containing a set of emissions.

        """
        self._mm = markov_model
        self._seq = sequence

    def _forward_recursion(self, cur_state, sequence_pos, forward_vars):
        """Calculate the forward recursion value (PRIVATE)."""
        raise NotImplementedError('Subclasses must implement')

    def forward_algorithm(self):
        """Calculate sequence probability using the forward algorithm.

        This implements the forward algorithm, as described on p57-58 of
        Durbin et al.

        Returns:
         - A dictionary containing the forward variables. This has keys of the
           form (state letter, position in the training sequence), and values
           containing the calculated forward variable.
         - The calculated probability of the sequence.

        """
        state_letters = self._mm.state_alphabet
        forward_var = {}
        forward_var[state_letters[0], -1] = 1
        for k in range(1, len(state_letters)):
            forward_var[state_letters[k], -1] = 0
        for i in range(len(self._seq.emissions)):
            for main_state in state_letters:
                forward_value = self._forward_recursion(main_state, i, forward_var)
                if forward_value is not None:
                    forward_var[main_state, i] = forward_value
        first_state = state_letters[0]
        seq_prob = 0
        for state_item in state_letters:
            forward_value = forward_var[state_item, len(self._seq.emissions) - 1]
            transition_value = self._mm.transition_prob[state_item, first_state]
            seq_prob += forward_value * transition_value
        return (forward_var, seq_prob)

    def _backward_recursion(self, cur_state, sequence_pos, forward_vars):
        """Calculate the backward recursion value (PRIVATE)."""
        raise NotImplementedError('Subclasses must implement')

    def backward_algorithm(self):
        """Calculate sequence probability using the backward algorithm.

        This implements the backward algorithm, as described on p58-59 of
        Durbin et al.

        Returns:
         - A dictionary containing the backwards variables. This has keys
           of the form (state letter, position in the training sequence),
           and values containing the calculated backward variable.

        """
        state_letters = self._mm.state_alphabet
        backward_var = {}
        first_letter = state_letters[0]
        for state in state_letters:
            backward_var[state, len(self._seq.emissions) - 1] = self._mm.transition_prob[state, state_letters[0]]
        all_indexes = list(range(len(self._seq.emissions) - 1))
        all_indexes.reverse()
        for i in all_indexes:
            for main_state in state_letters:
                backward_value = self._backward_recursion(main_state, i, backward_var)
                if backward_value is not None:
                    backward_var[main_state, i] = backward_value
        return backward_var