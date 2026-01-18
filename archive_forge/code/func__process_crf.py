from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
def _process_crf(process_tuple):
    """
    Extract the best beat sequence for a piece.

    This proxy function is necessary to process different intervals in parallel
    using the multiprocessing module.

    Parameters
    ----------
    process_tuple : tuple
        Tuple with (activations, dominant_interval, allowed deviation from the
        dominant interval per beat).

    Returns
    -------
    beats : numpy array
        Extracted beat positions [frames].
    log_prob : float
        Log probability of the beat sequence.

    """
    from .beats_crf import best_sequence
    return best_sequence(*process_tuple)