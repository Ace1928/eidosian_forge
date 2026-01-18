from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class BarTransitionModel(TransitionModel):
    """
    Transition model for bar tracking with a HMM.

    Within the beats of the bar the tempo stays the same; at beat boundaries
    transitions from one tempo (i.e. interval) to another following an
    exponential distribution are allowed.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    transition_lambda : float or list
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).
        None can be used to set the tempo change probability to 0.
        If a list is given, the individual values represent the lambdas for
        each transition into the beat at this index position.

    Notes
    -----
    Bars performing tempo changes only at bar boundaries (and not at the beat
    boundaries) must have set all but the first `transition_lambda` values to
    None, e.g. [100, None, None] for a bar with 3 beats.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """

    def __init__(self, state_space, transition_lambda):
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * state_space.num_beats
        if state_space.num_beats != len(transition_lambda):
            raise ValueError('length of `transition_lambda` must be equal to `num_beats` of `state_space`.')
        self.state_space = state_space
        self.transition_lambda = transition_lambda
        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=np.float)
        for beat in range(state_space.num_beats):
            to_states = state_space.first_states[beat]
            from_states = state_space.last_states[beat - 1]
            from_int = state_space.state_intervals[from_states]
            to_int = state_space.state_intervals[to_states]
            prob = exponential_transition(from_int, to_int, transition_lambda[beat])
            from_prob, to_prob = np.nonzero(prob)
            states = np.hstack((states, to_states[to_prob]))
            prev_states = np.hstack((prev_states, from_states[from_prob]))
            probabilities = np.hstack((probabilities, prob[prob != 0]))
        transitions = self.make_sparse(states, prev_states, probabilities)
        super(BarTransitionModel, self).__init__(*transitions)