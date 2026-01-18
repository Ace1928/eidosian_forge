import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
class MarkovModelBuilder:
    """Interface to build up a Markov Model.

    This class is designed to try to separate the task of specifying the
    Markov Model from the actual model itself. This is in hopes of making
    the actual Markov Model classes smaller.

    So, this builder class should be used to create Markov models instead
    of trying to initiate a Markov Model directly.
    """
    DEFAULT_PSEUDO = 1

    def __init__(self, state_alphabet, emission_alphabet):
        """Initialize a builder to create Markov Models.

        Arguments:
         - state_alphabet -- An iterable (e.g., tuple or list) containing
           all of the letters that can appear in the states
         - emission_alphabet -- An iterable (e.g., tuple or list) containing
           all of the letters for states that can be emitted by the HMM.

        """
        self._state_alphabet = tuple(state_alphabet)
        self._emission_alphabet = tuple(emission_alphabet)
        self.initial_prob = {}
        self.transition_prob = {}
        self.emission_prob = self._all_blank(state_alphabet, emission_alphabet)
        self.transition_pseudo = {}
        self.emission_pseudo = self._all_pseudo(state_alphabet, emission_alphabet)

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

    def _all_pseudo(self, first_alphabet, second_alphabet):
        """Return a dictionary with all counts set to a default value (PRIVATE).

        This takes the letters in first alphabet and second alphabet and
        creates a dictionary with keys of two tuples organized as:
        (letter of first alphabet, letter of second alphabet). The values
        are all set to the value of the class attribute DEFAULT_PSEUDO.
        """
        all_counts = {}
        for first_state in first_alphabet:
            for second_state in second_alphabet:
                all_counts[first_state, second_state] = self.DEFAULT_PSEUDO
        return all_counts

    def get_markov_model(self):
        """Return the markov model corresponding with the current parameters.

        Each markov model returned by a call to this function is unique
        (ie. they don't influence each other).
        """
        if not self.initial_prob:
            raise Exception('set_initial_probabilities must be called to fully initialize the Markov model')
        initial_prob = copy.deepcopy(self.initial_prob)
        transition_prob = copy.deepcopy(self.transition_prob)
        emission_prob = copy.deepcopy(self.emission_prob)
        transition_pseudo = copy.deepcopy(self.transition_pseudo)
        emission_pseudo = copy.deepcopy(self.emission_pseudo)
        return HiddenMarkovModel(self._state_alphabet, self._emission_alphabet, initial_prob, transition_prob, emission_prob, transition_pseudo, emission_pseudo)

    def set_initial_probabilities(self, initial_prob):
        """Set initial state probabilities.

        initial_prob is a dictionary mapping states to probabilities.
        Suppose, for example, that the state alphabet is ('A', 'B'). Call
        set_initial_prob({'A': 1}) to guarantee that the initial
        state will be 'A'. Call set_initial_prob({'A': 0.5, 'B': 0.5})
        to make each initial state equally probable.

        This method must now be called in order to use the Markov model
        because the calculation of initial probabilities has changed
        incompatibly; the previous calculation was incorrect.

        If initial probabilities are set for all states, then they should add up
        to 1. Otherwise the sum should be <= 1. The residual probability is
        divided up evenly between all the states for which the initial
        probability has not been set. For example, calling
        set_initial_prob({}) results in P('A') = 0.5 and P('B') = 0.5,
        for the above example.
        """
        self.initial_prob = copy.copy(initial_prob)
        for state in initial_prob:
            if state not in self._state_alphabet:
                raise ValueError(f'State {state} was not found in the sequence alphabet')
        num_states_not_set = len(self._state_alphabet) - len(self.initial_prob)
        if num_states_not_set < 0:
            raise Exception("Initial probabilities can't exceed # of states")
        prob_sum = sum(self.initial_prob.values())
        if prob_sum > 1.0:
            raise Exception('Total initial probability cannot exceed 1.0')
        if num_states_not_set > 0:
            prob = (1.0 - prob_sum) / num_states_not_set
            for state in self._state_alphabet:
                if state not in self.initial_prob:
                    self.initial_prob[state] = prob

    def set_equal_probabilities(self):
        """Reset all probabilities to be an average value.

        Resets the values of all initial probabilities and all allowed
        transitions and all allowed emissions to be equal to 1 divided by the
        number of possible elements.

        This is useful if you just want to initialize a Markov Model to
        starting values (ie. if you have no prior notions of what the
        probabilities should be -- or if you are just feeling too lazy
        to calculate them :-).

        Warning 1 -- this will reset all currently set probabilities.

        Warning 2 -- This just sets all probabilities for transitions and
        emissions to total up to 1, so it doesn't ensure that the sum of
        each set of transitions adds up to 1.
        """
        new_initial_prob = 1.0 / len(self.transition_prob)
        for state in self._state_alphabet:
            self.initial_prob[state] = new_initial_prob
        new_trans_prob = 1.0 / len(self.transition_prob)
        for key in self.transition_prob:
            self.transition_prob[key] = new_trans_prob
        new_emission_prob = 1.0 / len(self.emission_prob)
        for key in self.emission_prob:
            self.emission_prob[key] = new_emission_prob

    def set_random_initial_probabilities(self):
        """Set all initial state probabilities to a randomly generated distribution.

        Returns the dictionary containing the initial probabilities.
        """
        initial_freqs = _gen_random_array(len(self._state_alphabet))
        for state in self._state_alphabet:
            self.initial_prob[state] = initial_freqs.pop()
        return self.initial_prob

    def set_random_transition_probabilities(self):
        """Set all allowed transition probabilities to a randomly generated distribution.

        Returns the dictionary containing the transition probabilities.
        """
        if not self.transition_prob:
            raise Exception('No transitions have been allowed yet. Allow some or all transitions by calling allow_transition or allow_all_transitions first.')
        transitions_from = _calculate_from_transitions(self.transition_prob)
        for from_state in transitions_from:
            freqs = _gen_random_array(len(transitions_from[from_state]))
            for to_state in transitions_from[from_state]:
                self.transition_prob[from_state, to_state] = freqs.pop()
        return self.transition_prob

    def set_random_emission_probabilities(self):
        """Set all allowed emission probabilities to a randomly generated distribution.

        Returns the dictionary containing the emission probabilities.
        """
        if not self.emission_prob:
            raise Exception('No emissions have been allowed yet. Allow some or all emissions.')
        emissions = _calculate_emissions(self.emission_prob)
        for state in emissions:
            freqs = _gen_random_array(len(emissions[state]))
            for symbol in emissions[state]:
                self.emission_prob[state, symbol] = freqs.pop()
        return self.emission_prob

    def set_random_probabilities(self):
        """Set all probabilities to randomly generated numbers.

        Resets probabilities of all initial states, transitions, and
        emissions to random values.
        """
        self.set_random_initial_probabilities()
        self.set_random_transition_probabilities()
        self.set_random_emission_probabilities()

    def allow_all_transitions(self):
        """Create transitions between all states.

        By default all transitions within the alphabet are disallowed;
        this is a convenience function to change this to allow all
        possible transitions.
        """
        all_probs = self._all_blank(self._state_alphabet, self._state_alphabet)
        all_pseudo = self._all_pseudo(self._state_alphabet, self._state_alphabet)
        for set_key in self.transition_prob:
            all_probs[set_key] = self.transition_prob[set_key]
        for set_key in self.transition_pseudo:
            all_pseudo[set_key] = self.transition_pseudo[set_key]
        self.transition_prob = all_probs
        self.transition_pseudo = all_pseudo

    def allow_transition(self, from_state, to_state, probability=None, pseudocount=None):
        """Set a transition as being possible between the two states.

        probability and pseudocount are optional arguments
        specifying the probabilities and pseudo counts for the transition.
        If these are not supplied, then the values are set to the
        default values.

        Raises:
        KeyError -- if the two states already have an allowed transition.

        """
        for state in [from_state, to_state]:
            if state not in self._state_alphabet:
                raise ValueError(f'State {state} was not found in the sequence alphabet')
        if (from_state, to_state) not in self.transition_prob and (from_state, to_state) not in self.transition_pseudo:
            if probability is None:
                probability = 0
            self.transition_prob[from_state, to_state] = probability
            if pseudocount is None:
                pseudocount = self.DEFAULT_PSEUDO
            self.transition_pseudo[from_state, to_state] = pseudocount
        else:
            raise KeyError(f'Transition from {from_state} to {to_state} is already allowed.')

    def destroy_transition(self, from_state, to_state):
        """Restrict transitions between the two states.

        Raises:
        KeyError if the transition is not currently allowed.

        """
        try:
            del self.transition_prob[from_state, to_state]
            del self.transition_pseudo[from_state, to_state]
        except KeyError:
            raise KeyError(f'Transition from {from_state} to {to_state} is already disallowed.')

    def set_transition_score(self, from_state, to_state, probability):
        """Set the probability of a transition between two states.

        Raises:
        KeyError if the transition is not allowed.

        """
        if (from_state, to_state) in self.transition_prob:
            self.transition_prob[from_state, to_state] = probability
        else:
            raise KeyError(f'Transition from {from_state} to {to_state} is not allowed.')

    def set_transition_pseudocount(self, from_state, to_state, count):
        """Set the default pseudocount for a transition.

        To avoid computational problems, it is helpful to be able to
        set a 'default' pseudocount to start with for estimating
        transition and emission probabilities (see p62 in Durbin et al
        for more discussion on this. By default, all transitions have
        a pseudocount of 1.

        Raises:
        KeyError if the transition is not allowed.

        """
        if (from_state, to_state) in self.transition_pseudo:
            self.transition_pseudo[from_state, to_state] = count
        else:
            raise KeyError(f'Transition from {from_state} to {to_state} is not allowed.')

    def set_emission_score(self, seq_state, emission_state, probability):
        """Set the probability of a emission from a particular state.

        Raises:
        KeyError if the emission from the given state is not allowed.

        """
        if (seq_state, emission_state) in self.emission_prob:
            self.emission_prob[seq_state, emission_state] = probability
        else:
            raise KeyError(f'Emission of {emission_state} from {seq_state} is not allowed.')

    def set_emission_pseudocount(self, seq_state, emission_state, count):
        """Set the default pseudocount for an emission.

        To avoid computational problems, it is helpful to be able to
        set a 'default' pseudocount to start with for estimating
        transition and emission probabilities (see p62 in Durbin et al
        for more discussion on this. By default, all emissions have
        a pseudocount of 1.

        Raises:
        KeyError if the emission from the given state is not allowed.

        """
        if (seq_state, emission_state) in self.emission_pseudo:
            self.emission_pseudo[seq_state, emission_state] = count
        else:
            raise KeyError(f'Emission of {emission_state} from {seq_state} is not allowed.')