from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.ml.hmm import ObservationModel, TransitionModel
class RNNDownBeatTrackingObservationModel(ObservationModel):
    """
    Observation model for downbeat tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    observation_lambda : int
        Split each (down-)beat period into `observation_lambda` parts, the
        first representing (down-)beat states and the remaining non-beat
        states.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.

    """

    def __init__(self, state_space, observation_lambda):
        self.observation_lambda = observation_lambda
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        border = 1.0 / observation_lambda
        pointers[state_space.state_positions % 1 < border] = 1
        pointers[state_space.state_positions < border] = 2
        super(RNNDownBeatTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Compute the log densities of the observations.

        Parameters
        ----------
        observations : numpy array, shape (N, 2)
            Observations (i.e. 2D activations of a RNN, the columns represent
            'beat' and 'downbeat' probabilities)

        Returns
        -------
        numpy array, shape (N, 3)
            Log densities of the observations, the columns represent the
            observation log probability densities for no-beats, beats and
            downbeats.

        """
        log_densities = np.empty((len(observations), 3), dtype=np.float)
        log_densities[:, 0] = np.log((1.0 - np.sum(observations, axis=1)) / (self.observation_lambda - 1))
        log_densities[:, 1] = np.log(observations[:, 0])
        log_densities[:, 2] = np.log(observations[:, 1])
        return log_densities