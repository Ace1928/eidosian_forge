import copy
import math
import random
from collections import defaultdict
import warnings
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning
class HiddenMarkovModel:
    """Represent a hidden markov model that can be used for state estimation."""

    def __init__(self, state_alphabet, emission_alphabet, initial_prob, transition_prob, emission_prob, transition_pseudo, emission_pseudo):
        """Initialize a Markov Model.

        Note: You should use the MarkovModelBuilder class instead of
        initiating this class directly.

        Arguments:
         - state_alphabet -- A tuple containing all of the letters that can
           appear in the states.
         - emission_alphabet -- A tuple containing all of the letters for
           states that can be emitted by the HMM.
         - initial_prob - A dictionary of initial probabilities for all states.
         - transition_prob -- A dictionary of transition probabilities for all
           possible transitions in the sequence.
         - emission_prob -- A dictionary of emission probabilities for all
           possible emissions from the sequence states.
         - transition_pseudo -- Pseudo-counts to be used for the transitions,
           when counting for purposes of estimating transition probabilities.
         - emission_pseudo -- Pseudo-counts to be used for the emissions,
           when counting for purposes of estimating emission probabilities.

        """
        self.state_alphabet = state_alphabet
        self.emission_alphabet = emission_alphabet
        self.initial_prob = initial_prob
        self._transition_pseudo = transition_pseudo
        self._emission_pseudo = emission_pseudo
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self._transitions_from = _calculate_from_transitions(self.transition_prob)
        self._transitions_to = _calculate_to_transitions(self.transition_prob)

    def get_blank_transitions(self):
        """Get the default transitions for the model.

        Returns a dictionary of all of the default transitions between any
        two letters in the sequence alphabet. The dictionary is structured
        with keys as (letter1, letter2) and values as the starting number
        of transitions.
        """
        return self._transition_pseudo

    def get_blank_emissions(self):
        """Get the starting default emmissions for each sequence.

        This returns a dictionary of the default emmissions for each
        letter. The dictionary is structured with keys as
        (seq_letter, emmission_letter) and values as the starting number
        of emmissions.
        """
        return self._emission_pseudo

    def transitions_from(self, state_letter):
        """Get all destination states which can transition from source state_letter.

        This returns all letters which the given state_letter can transition
        to, i.e. all the destination states reachable from state_letter.

        An empty list is returned if state_letter has no outgoing transitions.
        """
        if state_letter in self._transitions_from:
            return self._transitions_from[state_letter]
        else:
            return []

    def transitions_to(self, state_letter):
        """Get all source states which can transition to destination state_letter.

        This returns all letters which the given state_letter is reachable
        from, i.e. all the source states which can reach state_later

        An empty list is returned if state_letter is unreachable.
        """
        if state_letter in self._transitions_to:
            return self._transitions_to[state_letter]
        else:
            return []

    def viterbi(self, sequence, state_alphabet):
        """Calculate the most probable state path using the Viterbi algorithm.

        This implements the Viterbi algorithm (see pgs 55-57 in Durbin et
        al for a full explanation -- this is where I took my implementation
        ideas from), to allow decoding of the state path, given a sequence
        of emissions.

        Arguments:
         - sequence -- A Seq object with the emission sequence that we
           want to decode.
         - state_alphabet -- An iterable (e.g., tuple or list) containing
           all of the letters that can appear in the states

        """
        log_initial = self._log_transform(self.initial_prob)
        log_trans = self._log_transform(self.transition_prob)
        log_emission = self._log_transform(self.emission_prob)
        viterbi_probs = {}
        pred_state_seq = {}
        for i in range(len(sequence)):
            for cur_state in state_alphabet:
                emission_part = log_emission[cur_state, sequence[i]]
                max_prob = 0
                if i == 0:
                    max_prob = log_initial[cur_state]
                else:
                    possible_state_probs = {}
                    for prev_state in self.transitions_to(cur_state):
                        trans_part = log_trans[prev_state, cur_state]
                        viterbi_part = viterbi_probs[prev_state, i - 1]
                        cur_prob = viterbi_part + trans_part
                        possible_state_probs[prev_state] = cur_prob
                    max_prob = max(possible_state_probs.values())
                viterbi_probs[cur_state, i] = emission_part + max_prob
                if i > 0:
                    for state in possible_state_probs:
                        if possible_state_probs[state] == max_prob:
                            pred_state_seq[i - 1, cur_state] = state
                            break
        all_probs = {}
        for state in state_alphabet:
            all_probs[state] = viterbi_probs[state, len(sequence) - 1]
        state_path_prob = max(all_probs.values())
        last_state = ''
        for state in all_probs:
            if all_probs[state] == state_path_prob:
                last_state = state
        assert last_state != '', "Didn't find the last state to trace from!"
        traceback_seq = []
        loop_seq = list(range(1, len(sequence)))
        loop_seq.reverse()
        state = last_state
        traceback_seq.append(state)
        for i in loop_seq:
            state = pred_state_seq[i - 1, state]
            traceback_seq.append(state)
        traceback_seq.reverse()
        traceback_seq = ''.join(traceback_seq)
        return (Seq(traceback_seq), state_path_prob)

    def _log_transform(self, probability):
        """Return log transform of the given probability dictionary (PRIVATE).

        When calculating the Viterbi equation, add logs of probabilities rather
        than multiplying probabilities, to avoid underflow errors. This method
        returns a new dictionary with the same keys as the given dictionary
        and log-transformed values.
        """
        log_prob = copy.copy(probability)
        for key in log_prob:
            prob = log_prob[key]
            if prob > 0:
                log_prob[key] = math.log(log_prob[key])
            else:
                log_prob[key] = -math.inf
        return log_prob