from __future__ import absolute_import, division, print_function
import numpy as np
from .onsets import OnsetPeakPickingProcessor, peak_picking
from ..processors import ParallelProcessor, SequentialProcessor
from ..utils import combine_events
class NotePeakPickingProcessor(OnsetPeakPickingProcessor):
    """
    This class implements the note peak-picking functionality.

    Parameters
    ----------
    threshold : float
        Threshold for peak-picking.
    smooth : float, optional
        Smooth the activation function over `smooth` seconds.
    pre_avg : float, optional
        Use `pre_avg` seconds past information for moving average.
    post_avg : float, optional
        Use `post_avg` seconds future information for moving average.
    pre_max : float, optional
        Use `pre_max` seconds past information for moving maximum.
    post_max : float, optional
        Use `post_max` seconds future information for moving maximum.
    combine : float, optional
        Only report one note per pitch within `combine` seconds.
    delay : float, optional
        Report the detected notes `delay` seconds delayed.
    online : bool, optional
        Use online peak-picking, i.e. no future information.
    fps : float, optional
        Frames per second used for conversion of timings.

    Returns
    -------
    notes : numpy array
        Detected notes [seconds, pitch].

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), `pre_avg` and
    `post_avg` should be set to 0.
    For peak picking of local maxima, set `pre_max` >= 1. / `fps` and
    `post_max` >= 1. / `fps`.
    For online peak picking, all `post_` parameters are set to 0.

    Examples
    --------
    Create a PeakPickingProcessor. The returned array represents the positions
    of the onsets in seconds, thus the expected sampling rate has to be given.

    >>> proc = NotePeakPickingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.notes.NotePeakPickingProcessor object at 0x...>

    Call this NotePeakPickingProcessor with the note activations from an
    RNNPianoNoteProcessor.

    >>> act = RNNPianoNoteProcessor()('tests/data/audio/stereo_sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([[ 0.14, 72.  ],
           [ 1.56, 41.  ],
           [ 3.37, 75.  ]])

    """
    FPS = 100
    THRESHOLD = 0.5
    SMOOTH = 0.0
    PRE_AVG = 0.0
    POST_AVG = 0.0
    PRE_MAX = 0.0
    POST_MAX = 0.0
    COMBINE = 0.03
    DELAY = 0.0
    ONLINE = False

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG, post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX, combine=COMBINE, delay=DELAY, online=ONLINE, fps=FPS, **kwargs):
        super(NotePeakPickingProcessor, self).__init__(threshold=threshold, smooth=smooth, pre_avg=pre_avg, post_avg=post_avg, pre_max=pre_max, post_max=post_max, combine=combine, delay=delay, online=online, fps=fps)

    def process(self, activations, **kwargs):
        """
        Detect the notes in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Note activation function.

        Returns
        -------
        onsets : numpy array
            Detected notes [seconds, pitches].

        """
        timings = np.array([self.smooth, self.pre_avg, self.post_avg, self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        notes = peak_picking(activations, self.threshold, *timings)
        onsets = notes[0].astype(np.float) / self.fps
        pitches = notes[1] + 21
        if self.delay:
            onsets += self.delay
        if self.combine > 0:
            notes = []
            for pitch in np.unique(pitches):
                onsets_ = onsets[pitches == pitch]
                onsets_ = combine_events(onsets_, self.combine, 'left')
                notes.extend(list(zip(onsets_, [pitch] * len(onsets_))))
        else:
            notes = list(zip(onsets, pitches))
        return np.asarray(sorted(notes))