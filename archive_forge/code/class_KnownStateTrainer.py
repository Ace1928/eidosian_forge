import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
class KnownStateTrainer(AbstractTrainer):
    """Estimate probabilities with known state sequences.

    This should be used for direct estimation of emission and transition
    probabilities when both the state path and emission sequence are
    known for the training examples.
    """

    def __init__(self, markov_model):
        """Initialize the class."""
        AbstractTrainer.__init__(self, markov_model)

    def train(self, training_seqs):
        """Estimate the Markov Model parameters with known state paths.

        This trainer requires that both the state and the emissions are
        known for all of the training sequences in the list of
        TrainingSequence objects.
        This training will then count all of the transitions and emissions,
        and use this to estimate the parameters of the model.
        """
        transition_counts = self._markov_model.get_blank_transitions()
        emission_counts = self._markov_model.get_blank_emissions()
        for training_seq in training_seqs:
            emission_counts = self._count_emissions(training_seq, emission_counts)
            transition_counts = self._count_transitions(training_seq.states, transition_counts)
        ml_transitions, ml_emissions = self.estimate_params(transition_counts, emission_counts)
        self._markov_model.transition_prob = ml_transitions
        self._markov_model.emission_prob = ml_emissions
        return self._markov_model

    def _count_emissions(self, training_seq, emission_counts):
        """Add emissions from the training sequence to the current counts (PRIVATE).

        Arguments:
         - training_seq -- A TrainingSequence with states and emissions
           to get the counts from
         - emission_counts -- The current emission counts to add to.

        """
        for index in range(len(training_seq.emissions)):
            cur_state = training_seq.states[index]
            cur_emission = training_seq.emissions[index]
            try:
                emission_counts[cur_state, cur_emission] += 1
            except KeyError:
                raise KeyError(f'Unexpected emission ({cur_state}, {cur_emission})')
        return emission_counts

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