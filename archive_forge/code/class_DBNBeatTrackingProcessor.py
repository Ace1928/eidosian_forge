from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
class DBNBeatTrackingProcessor(OnlineProcessor):
    """
    Beat tracking with RNNs and a dynamic Bayesian network (DBN) approximated
    by a Hidden Markov Model (HMM).

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo used for beat tracking [bpm].
    max_bpm : float, optional
        Maximum tempo used for beat tracking [bpm].
    num_tempi : int, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing.
    transition_lambda : float, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).
    observation_lambda : int, optional
        Split one beat period into `observation_lambda` parts, the first
        representing beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the observations before Viterbi decoding.
    correct : bool, optional
        Correct the beats (i.e. align them to the nearest peak of the beat
        activation function).
    fps : float, optional
        Frames per second.
    online : bool, optional
        Use the forward algorithm (instead of Viterbi) to decode the beats.

    Notes
    -----
    Instead of the originally proposed state space and transition model for
    the DBN [1]_, the more efficient version proposed in [2]_ is used.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.
    .. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a DBNBeatTrackingProcessor. The returned array represents the
    positions of the beats in seconds, thus the expected sampling rate has to
    be given.

    >>> proc = DBNBeatTrackingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.DBNBeatTrackingProcessor object at 0x...>

    Call this DBNBeatTrackingProcessor with the beat activation function
    returned by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.1 , 0.45, 0.8 , 1.12, 1.48, 1.8 , 2.15, 2.49])

    """
    MIN_BPM = 55.0
    MAX_BPM = 215.0
    NUM_TEMPI = None
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0
    CORRECT = True

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA, observation_lambda=OBSERVATION_LAMBDA, correct=CORRECT, threshold=THRESHOLD, fps=None, online=False, **kwargs):
        from .beats_hmm import BeatStateSpace, BeatTransitionModel, RNNBeatTrackingObservationModel
        from ..ml.hmm import HiddenMarkovModel
        min_interval = 60.0 * fps / max_bpm
        max_interval = 60.0 * fps / min_bpm
        self.st = BeatStateSpace(min_interval, max_interval, num_tempi)
        self.tm = BeatTransitionModel(self.st, transition_lambda)
        self.om = RNNBeatTrackingObservationModel(self.st, observation_lambda)
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        self.correct = correct
        self.threshold = threshold
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.online = online
        if self.online:
            self.visualize = kwargs.get('verbose', False)
            self.counter = 0
            self.beat_counter = 0
            self.strength = 0
            self.last_beat = 0
            self.tempo = 0

    def reset(self):
        """Reset the DBNBeatTrackingProcessor."""
        self.hmm.reset()
        self.counter = 0
        self.beat_counter = 0
        self.strength = 0
        self.last_beat = 0
        self.tempo = 0

    def process_offline(self, activations, **kwargs):
        """
        Detect the beats in the given activation function with Viterbi
        decoding.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
        beats = np.empty(0, dtype=np.int)
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        if not activations.any():
            return beats
        path, _ = self.hmm.viterbi(activations)
        if self.correct:
            beat_range = self.om.pointers[path]
            idx = np.nonzero(np.diff(beat_range))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) + left
                    beats = np.hstack((beats, peak))
        else:
            from scipy.signal import argrelmin
            beats = argrelmin(self.st.state_positions[path], mode='wrap')[0]
            beats = beats[self.om.pointers[path[beats]] == 1]
        return (beats + first) / float(self.fps)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the beats in the given activation function with the forward
        algorithm.

        Parameters
        ----------
        activations : numpy array
            Beat activation for a single frame.
        reset : bool, optional
            Reset the DBNBeatTrackingProcessor to its initial state before
            processing.

        Returns
        -------
        beats : numpy array
            Detected beat position [seconds].

        """
        if reset:
            self.reset()
        fwd = self.hmm.forward(activations, reset=reset)
        states = np.argmax(fwd, axis=1)
        beats = self.om.pointers[states] == 1
        positions = self.st.state_positions[states]
        if self.visualize and len(activations) == 1:
            beat_length = 80
            display = [' '] * beat_length
            display[int(positions * beat_length)] = '*'
            strength_length = 10
            self.strength = int(max(self.strength, activations * 10))
            display.append('| ')
            display.extend(['*'] * self.strength)
            display.extend([' '] * (strength_length - self.strength))
            if self.counter % 5 == 0:
                self.strength -= 1
            if beats:
                self.beat_counter = 3
            if self.beat_counter > 0:
                display.append('| X ')
            else:
                display.append('|   ')
            self.beat_counter -= 1
            display.append('| %5.1f | ' % self.tempo)
            sys.stderr.write('\r%s' % ''.join(display))
            sys.stderr.flush()
        beats_ = []
        for frame in np.nonzero(beats)[0]:
            cur_beat = (frame + self.counter) / float(self.fps)
            next_beat = self.last_beat + 60.0 / self.max_bpm
            if cur_beat >= next_beat:
                self.tempo = 60.0 / (cur_beat - self.last_beat)
                self.last_beat = cur_beat
                beats_.append(cur_beat)
        self.counter += len(activations)
        return np.array(beats_)
    process_forward = process_online
    process_viterbi = process_offline

    @staticmethod
    def add_arguments(parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA, observation_lambda=OBSERVATION_LAMBDA, threshold=THRESHOLD, correct=CORRECT):
        """
        Add DBN related arguments to an existing parser object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        min_bpm : float, optional
            Minimum tempo used for beat tracking [bpm].
        max_bpm : float, optional
            Maximum tempo used for beat tracking [bpm].
        num_tempi : int, optional
            Number of tempi to model; if set, limit the number of tempi and use
            a log spacing, otherwise a linear spacing.
        transition_lambda : float, optional
            Lambda for the exponential tempo change distribution (higher values
            prefer a constant tempo over a tempo change from one beat to the
            next one).
        observation_lambda : float, optional
            Split one beat period into `observation_lambda` parts, the first
            representing beat states and the remaining non-beat states.
        threshold : float, optional
            Threshold the observations before Viterbi decoding.
        correct : bool, optional
            Correct the beats (i.e. align them to the nearest peak of the beat
            activation function).

        Returns
        -------
        parser_group : argparse argument group
            DBN beat tracking argument parser group

        """
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--min_bpm', action='store', type=float, default=min_bpm, help='minimum tempo [bpm, default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float, default=max_bpm, help='maximum tempo [bpm,  default=%(default).2f]')
        g.add_argument('--num_tempi', action='store', type=int, default=num_tempi, help='limit the number of tempi; if set, align the tempi with a log spacing, otherwise linearly')
        g.add_argument('--transition_lambda', action='store', type=float, default=transition_lambda, help='lambda of the tempo transition distribution; higher values prefer a constant tempo over a tempo change from one beat to the next one [default=%(default).1f]')
        g.add_argument('--observation_lambda', action='store', type=float, default=observation_lambda, help='split one beat period into N parts, the first representing beat states and the remaining non-beat states [default=%(default)i]')
        g.add_argument('-t', dest='threshold', action='store', type=float, default=threshold, help='threshold the observations before Viterbi decoding [default=%(default).2f]')
        if correct:
            g.add_argument('--no_correct', dest='correct', action='store_false', default=correct, help='do not correct the beat positions (i.e. do not align them to the nearest peak of the beat activation function)')
        else:
            g.add_argument('--correct', dest='correct', action='store_true', default=correct, help='correct the beat positions (i.e. align them to the nearest peak of the beat activationfunction)')
        return g