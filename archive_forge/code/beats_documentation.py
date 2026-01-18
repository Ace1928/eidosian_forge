from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,

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

        