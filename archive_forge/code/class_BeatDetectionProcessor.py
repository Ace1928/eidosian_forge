from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
class BeatDetectionProcessor(BeatTrackingProcessor):
    """
    Class for detecting beats according to the previously determined global
    tempo by iteratively aligning them around the estimated position [1]_.

    Parameters
    ----------
    look_aside : float
        Look this fraction of the estimated beat interval to each side of the
        assumed next beat position to look for the most likely position of the
        next beat.
    fps : float, optional
        Frames per second.

    Notes
    -----
    A constant tempo throughout the whole piece is assumed.

    Instead of the auto-correlation based method for tempo estimation proposed
    in [1]_, it uses a comb filter based method [2]_ per default. The behaviour
    can be controlled with the `tempo_method` parameter.

    See Also
    --------
    :class:`BeatTrackingProcessor`

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.
    .. [2] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Accurate Tempo Estimation based on Recurrent Neural Networks and
           Resonating Comb Filters",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a BeatDetectionProcessor. The returned array represents the
    positions of the beats in seconds, thus the expected sampling rate has to
    be given.

    >>> proc = BeatDetectionProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.BeatDetectionProcessor object at 0x...>

    Call this BeatDetectionProcessor with the beat activation function returned
    by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.11, 0.45, 0.79, 1.13, 1.47, 1.81, 2.15, 2.49])

    """
    LOOK_ASIDE = 0.2

    def __init__(self, look_aside=LOOK_ASIDE, fps=None, **kwargs):
        super(BeatDetectionProcessor, self).__init__(look_aside=look_aside, look_ahead=None, fps=fps, **kwargs)