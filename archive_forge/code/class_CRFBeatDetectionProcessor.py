from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
class CRFBeatDetectionProcessor(BeatTrackingProcessor):
    """
    Conditional Random Field Beat Detection.

    Tracks the beats according to the previously determined global tempo using
    a conditional random field (CRF) model.

    Parameters
    ----------
    interval_sigma : float, optional
        Allowed deviation from the dominant beat interval per beat.
    use_factors : bool, optional
        Use dominant interval multiplied by factors instead of intervals
        estimated by tempo estimator.
    num_intervals : int, optional
        Maximum number of estimated intervals to try.
    factors : list or numpy array, optional
        Factors of the dominant interval to try.

    References
    ----------
    .. [1] Filip Korzeniowski, Sebastian Böck and Gerhard Widmer,
           "Probabilistic Extraction of Beat Positions from a Beat Activation
           Function",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.

    Examples
    --------
    Create a CRFBeatDetectionProcessor. The returned array represents the
    positions of the beats in seconds, thus the expected sampling rate has to
    be given.

    >>> proc = CRFBeatDetectionProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.CRFBeatDetectionProcessor object at 0x...>

    Call this BeatDetectionProcessor with the beat activation function returned
    by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.09, 0.79, 1.49])

    """
    INTERVAL_SIGMA = 0.18
    USE_FACTORS = False
    FACTORS = np.array([0.5, 0.67, 1.0, 1.5, 2.0])
    NUM_INTERVALS = 5
    MIN_BPM = 20
    MAX_BPM = 240
    ACT_SMOOTH = 0.09
    HIST_SMOOTH = 7

    def __init__(self, interval_sigma=INTERVAL_SIGMA, use_factors=USE_FACTORS, num_intervals=NUM_INTERVALS, factors=FACTORS, **kwargs):
        super(CRFBeatDetectionProcessor, self).__init__(**kwargs)
        self.interval_sigma = interval_sigma
        self.use_factors = use_factors
        self.num_intervals = num_intervals
        self.factors = factors
        num_threads = min(len(factors) if use_factors else num_intervals, kwargs.get('num_threads', 1))
        self.map = map
        if num_threads != 1:
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map

    def process(self, activations, **kwargs):
        """
        Detect the beats in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        numpy array
            Detected beat positions [seconds].

        """
        import itertools as it
        tempi = self.tempo_estimator.process(activations)
        intervals = self.fps * 60.0 / tempi[:, 0]
        if self.use_factors:
            possible_intervals = [int(intervals[0] * f) for f in self.factors]
            possible_intervals = [i for i in possible_intervals if self.tempo_estimator.max_interval >= i >= self.tempo_estimator.min_interval]
        else:
            possible_intervals = intervals[:self.num_intervals]
        possible_intervals.sort()
        possible_intervals = [int(i) for i in possible_intervals[::-1]]
        act_smooth = int(self.fps * self.tempo_estimator.act_smooth)
        activations = smooth_signal(activations, act_smooth)
        contiguous_act = np.ascontiguousarray(activations, dtype=np.float32)
        results = list(self.map(_process_crf, zip(it.repeat(contiguous_act), possible_intervals, it.repeat(self.interval_sigma))))
        normalized_seq_probabilities = np.array([r[1] / r[0].shape[0] for r in results])
        best_seq = results[normalized_seq_probabilities.argmax()][0]
        return best_seq.astype(np.float) / self.fps

    @staticmethod
    def add_arguments(parser, interval_sigma=INTERVAL_SIGMA, use_factors=USE_FACTORS, num_intervals=NUM_INTERVALS, factors=FACTORS):
        """
        Add CRFBeatDetection related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        interval_sigma : float, optional
            allowed deviation from the dominant beat interval per beat
        use_factors : bool, optional
            use dominant interval multiplied by factors instead of intervals
            estimated by tempo estimator
        num_intervals : int, optional
            max number of estimated intervals to try
        factors : list or numpy array, optional
            factors of the dominant interval to try

        Returns
        -------
        parser_group : argparse argument group
            CRF beat tracking argument parser group.

        """
        from ..utils import OverrideDefaultListAction
        g = parser.add_argument_group('conditional random field arguments')
        g.add_argument('--interval_sigma', action='store', type=float, default=interval_sigma, help='allowed deviation from the dominant interval [default=%(default).2f]')
        g.add_argument('--use_factors', action='store_true', default=use_factors, help='use dominant interval multiplied with factors instead of multiple estimated intervals [default=%(default)s]')
        g.add_argument('--num_intervals', action='store', type=int, default=num_intervals, dest='num_intervals', help='number of estimated intervals to try [default=%(default)s]')
        g.add_argument('--factors', action=OverrideDefaultListAction, default=factors, type=float, sep=',', help='(comma separated) list with factors of dominant interval to try [default=%(default)s]')
        return g