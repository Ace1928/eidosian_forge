from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
class PatternTrackingProcessor(Processor):
    """
    Pattern tracking with a dynamic Bayesian network (DBN) approximated by a
    Hidden Markov Model (HMM).

    Parameters
    ----------
    pattern_files : list
        List of files with the patterns (including the fitted GMMs and
        information about the number of beats).
    min_bpm : list, optional
        Minimum tempi used for pattern tracking [bpm].
    max_bpm : list, optional
        Maximum tempi used for pattern tracking [bpm].
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacings, otherwise a linear spacings.
    transition_lambda : float or list, optional
        Lambdas for the exponential tempo change distributions (higher values
        prefer constant tempi from one beat to the next one).
    fps : float, optional
        Frames per second.

    Notes
    -----
    `min_bpm`, `max_bpm`, `num_tempo_states`, and `transition_lambda` must
    contain as many items as rhythmic patterns are modeled (i.e. length of
    `pattern_files`).
    If a single value is given for `num_tempo_states` and `transition_lambda`,
    this value is used for all rhythmic patterns.

    Instead of the originally proposed state space and transition model for
    the DBN [1]_, the more efficient version proposed in [2]_ is used.

    References
    ----------
    .. [1] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
           Audio",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.
    .. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a PatternTrackingProcessor from the given pattern files. These
    pattern files include fitted GMMs for the observation model of the HMM.
    The returned array represents the positions of the beats and their position
    inside the bar. The position is given in seconds, thus the expected
    sampling rate is needed. The position inside the bar follows the natural
    counting and starts at 1.

    >>> from madmom.models import PATTERNS_BALLROOM
    >>> proc = PatternTrackingProcessor(PATTERNS_BALLROOM, fps=50)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.downbeats.PatternTrackingProcessor object at 0x...>

    Call this PatternTrackingProcessor with a multi-band spectrogram to obtain
    the beat and downbeat positions. The parameters of the spectrogram have to
    correspond to those used to fit the GMMs.

    >>> from madmom.audio.spectrogram import LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor, MultiBandSpectrogramProcessor
    >>> from madmom.processors import SequentialProcessor
    >>> log = LogarithmicSpectrogramProcessor()
    >>> diff = SpectrogramDifferenceProcessor(positive_diffs=True)
    >>> mb = MultiBandSpectrogramProcessor(crossover_frequencies=[270])
    >>> pre_proc = SequentialProcessor([log, diff, mb])

    >>> act = pre_proc('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([[0.82, 4.  ],
           [1.78, 1.  ],
           ...,
           [3.7 , 3.  ],
           [4.66, 4.  ]])
    """
    MIN_BPM = (55, 60)
    MAX_BPM = (205, 225)
    NUM_TEMPI = None
    TRANSITION_LAMBDA = 100

    def __init__(self, pattern_files, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA, fps=None, **kwargs):
        import pickle
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        if len(min_bpm) != len(pattern_files):
            min_bpm = np.repeat(min_bpm, len(pattern_files))
        if len(max_bpm) != len(pattern_files):
            max_bpm = np.repeat(max_bpm, len(pattern_files))
        if len(num_tempi) != len(pattern_files):
            num_tempi = np.repeat(num_tempi, len(pattern_files))
        if len(transition_lambda) != len(pattern_files):
            transition_lambda = np.repeat(transition_lambda, len(pattern_files))
        if not len(min_bpm) == len(max_bpm) == len(num_tempi) == len(transition_lambda) == len(pattern_files):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi` and `transition_lambda` must have the same length as number of patterns.')
        self.fps = fps
        self.num_beats = []
        min_interval = 60.0 * self.fps / np.asarray(max_bpm)
        max_interval = 60.0 * self.fps / np.asarray(min_bpm)
        state_spaces = []
        transition_models = []
        gmms = []
        if not pattern_files:
            raise ValueError('at least one rhythmical pattern must be given.')
        for p, pattern_file in enumerate(pattern_files):
            with open(pattern_file, 'rb') as f:
                try:
                    pattern = pickle.load(f, encoding='latin1')
                except TypeError:
                    pattern = pickle.load(f)
            gmms.append(pattern['gmms'])
            num_beats = pattern['num_beats']
            self.num_beats.append(num_beats)
            state_space = BarStateSpace(num_beats, min_interval[p], max_interval[p], num_tempi[p])
            transition_model = BarTransitionModel(state_space, transition_lambda[p])
            state_spaces.append(state_space)
            transition_models.append(transition_model)
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(transition_models)
        self.om = GMMPatternTrackingObservationModel(gmms, self.st)
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)

    def process(self, features, **kwargs):
        """
        Detect the (down-)beats given the features.

        Parameters
        ----------
        features : numpy array
            Multi-band spectral features.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """
        path, _ = self.hmm.viterbi(features)
        positions = self.st.state_positions[path]
        beat_numbers = positions.astype(int) + 1
        beat_positions = np.nonzero(np.diff(beat_numbers))[0] + 1
        return np.vstack((beat_positions / float(self.fps), beat_numbers[beat_positions])).T

    @staticmethod
    def add_arguments(parser, pattern_files=None, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA):
        """
        Add DBN related arguments for pattern tracking to an existing parser
        object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        pattern_files : list
            Load the patterns from these files.
        min_bpm : list, optional
            Minimum tempi used for beat tracking [bpm].
        max_bpm : list, optional
            Maximum tempi used for beat tracking [bpm].
        num_tempi : int or list, optional
            Number of tempi to model; if set, limit the number of states and
            use log spacings, otherwise a linear spacings.
        transition_lambda : float or list, optional
            Lambdas for the exponential tempo change distribution (higher
            values prefer constant tempi from one beat to the next one).

        Returns
        -------
        parser_group : argparse argument group
            Pattern tracking argument parser group

        Notes
        -----
        `pattern_files`, `min_bpm`, `max_bpm`, `num_tempi`, and
        `transition_lambda` must have the same number of items.

        """
        from ..utils import OverrideDefaultListAction
        if pattern_files is not None:
            g = parser.add_argument_group('GMM arguments')
            g.add_argument('--pattern_files', action=OverrideDefaultListAction, default=pattern_files, help='load the patterns (with the fitted GMMs) from these files (comma separated list)')
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--min_bpm', action=OverrideDefaultListAction, default=min_bpm, type=float, sep=',', help='minimum tempo (comma separated list with one value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--max_bpm', action=OverrideDefaultListAction, default=max_bpm, type=float, sep=',', help='maximum tempo (comma separated list with one value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--num_tempi', action=OverrideDefaultListAction, default=num_tempi, type=int, sep=',', help='limit the number of tempi; if set, align the tempi with log spacings, otherwise linearly (comma separated list with one value per pattern) [default=%(default)s]')
        g.add_argument('--transition_lambda', action=OverrideDefaultListAction, default=transition_lambda, type=float, sep=',', help='lambda of the tempo transition distribution; higher values prefer a constant tempo over a tempo change from one bar to the next one (comma separated list with one value per pattern) [default=%(default)s]')
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False, help='output only the downbeats')
        return g