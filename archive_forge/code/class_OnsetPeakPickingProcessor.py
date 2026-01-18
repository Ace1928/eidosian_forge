from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
class OnsetPeakPickingProcessor(OnlineProcessor):
    """
    This class implements the onset peak-picking functionality.
    It transparently converts the chosen values from seconds to frames.

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
        Only report one onset within `combine` seconds.
    delay : float, optional
        Report the detected onsets `delay` seconds delayed.
    online : bool, optional
        Use online peak-picking, i.e. no future information.
    fps : float, optional
        Frames per second used for conversion of timings.

    Returns
    -------
    onsets : numpy array
        Detected onsets [seconds].

    Notes
    -----
    If no moving average is needed (e.g. the activations are independent of
    the signal's level as for neural network activations), `pre_avg` and
    `post_avg` should be set to 0.
    For peak picking of local maxima, set `pre_max` >= 1. / `fps` and
    `post_max` >= 1. / `fps`.
    For online peak picking, all `post_` parameters are set to 0.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Markus Schedl,
           "Evaluating the Online Capabilities of Onset Detection Methods",
           Proceedings of the 13th International Society for Music Information
           Retrieval Conference (ISMIR), 2012.

    Examples
    --------
    Create a PeakPickingProcessor. The returned array represents the positions
    of the onsets in seconds, thus the expected sampling rate has to be given.

    >>> proc = OnsetPeakPickingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.onsets.OnsetPeakPickingProcessor object at 0x...>

    Call this OnsetPeakPickingProcessor with the onset activation function from
    an RNNOnsetProcessor to obtain the onset positions.

    >>> act = RNNOnsetProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([0.09, 0.29, 0.45, ..., 2.34, 2.49, 2.67])

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
        super(OnsetPeakPickingProcessor, self).__init__(online=online)
        if self.online:
            smooth = 0
            post_avg = 0
            post_max = 0
            self.buffer = None
            self.counter = 0
            self.last_onset = None
        self.threshold = threshold
        self.smooth = smooth
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.pre_max = pre_max
        self.post_max = post_max
        self.combine = combine
        self.delay = delay
        self.fps = fps

    def reset(self):
        """Reset OnsetPeakPickingProcessor."""
        self.buffer = None
        self.counter = 0
        self.last_onset = None

    def process_offline(self, activations, **kwargs):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        onsets : numpy array
            Detected onsets [seconds].

        """
        timings = np.array([self.smooth, self.pre_avg, self.post_avg, self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        onsets = peak_picking(activations, self.threshold, *timings)
        onsets = onsets.astype(np.float) / self.fps
        if self.delay:
            onsets += self.delay
        if self.combine:
            onsets = combine_events(onsets, self.combine, 'left')
        return np.asarray(onsets)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the onsets in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.
        reset : bool, optional
            Reset the processor to its initial state before processing.

        Returns
        -------
        onsets : numpy array
            Detected onsets [seconds].

        """
        if self.buffer is None or reset:
            self.reset()
            init = np.zeros(int(np.round(self.pre_max * self.fps)))
            buffer = np.insert(activations, 0, init, axis=0)
            self.counter = -len(init)
            self.buffer = BufferProcessor(init=buffer)
        else:
            buffer = self.buffer(activations)
        timings = np.array([self.smooth, self.pre_avg, self.post_avg, self.pre_max, self.post_max]) * self.fps
        timings = np.round(timings).astype(int)
        peaks = peak_picking(buffer, self.threshold, *timings)
        onsets = (self.counter + peaks) / float(self.fps)
        self.counter += len(activations)
        if self.delay:
            raise ValueError('delay not supported yet in online mode')
        if self.combine and onsets.any():
            start = 0
            if self.last_onset is not None:
                onsets = np.append(self.last_onset, onsets)
                start = 1
            onsets = combine_events(onsets, self.combine, 'left')
            if onsets[-1] != self.last_onset:
                self.last_onset = onsets[-1]
                onsets = onsets[start:]
            else:
                onsets = np.empty(0)
        return onsets
    process_sequence = process_offline

    @staticmethod
    def add_arguments(parser, threshold=THRESHOLD, smooth=None, pre_avg=None, post_avg=None, pre_max=None, post_max=None, combine=COMBINE, delay=DELAY):
        """
        Add onset peak-picking related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
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
            Only report one onset within `combine` seconds.
        delay : float, optional
            Report the detected onsets `delay` seconds delayed.

        Returns
        -------
        parser_group : argparse argument group
            Onset peak-picking argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        g = parser.add_argument_group('peak-picking arguments')
        g.add_argument('-t', dest='threshold', action='store', type=float, default=threshold, help='detection threshold [default=%(default).2f]')
        if smooth is not None:
            g.add_argument('--smooth', action='store', type=float, default=smooth, help='smooth the activation function over N seconds [default=%(default).2f]')
        if pre_avg is not None:
            g.add_argument('--pre_avg', action='store', type=float, default=pre_avg, help='build average over N previous seconds [default=%(default).2f]')
        if post_avg is not None:
            g.add_argument('--post_avg', action='store', type=float, default=post_avg, help='build average over N following seconds [default=%(default).2f]')
        if pre_max is not None:
            g.add_argument('--pre_max', action='store', type=float, default=pre_max, help='search maximum over N previous seconds [default=%(default).2f]')
        if post_max is not None:
            g.add_argument('--post_max', action='store', type=float, default=post_max, help='search maximum over N following seconds [default=%(default).2f]')
        if combine is not None:
            g.add_argument('--combine', action='store', type=float, default=combine, help='combine events within N seconds [default=%(default).2f]')
        if delay is not None:
            g.add_argument('--delay', action='store', type=float, default=delay, help='report the events N seconds delayed [default=%(default)i]')
        return g