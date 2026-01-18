from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
class DBNTempoHistogramProcessor(TempoHistogramProcessor):
    """
    Create a tempo histogram with a dynamic Bayesian network (DBN).

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo to detect [bpm].
    max_bpm : float, optional
        Maximum tempo to detect [bpm].
    hist_buffer : float
        Aggregate the tempo histogram over `hist_buffer` seconds.
    fps : float, optional
        Frames per second.
    online : bool, optional
        Operate in online (i.e. causal) mode.

    """

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, hist_buffer=HIST_BUFFER, fps=None, online=False, **kwargs):
        super(DBNTempoHistogramProcessor, self).__init__(min_bpm=min_bpm, max_bpm=max_bpm, hist_buffer=hist_buffer, fps=fps, online=online, **kwargs)
        from .beats import DBNBeatTrackingProcessor
        self.dbn = DBNBeatTrackingProcessor(min_bpm=self.min_bpm, max_bpm=self.max_bpm, fps=self.fps, online=online, **kwargs)

    def reset(self):
        """Reset DBN to initial state."""
        super(DBNTempoHistogramProcessor, self).reset()
        self.dbn.hmm.reset()

    def process_offline(self, activations, **kwargs):
        """
        Compute the histogram of the beat intervals with a DBN.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the beat interval histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].

        """
        path, _ = self.dbn.hmm.viterbi(activations.astype(np.float32))
        intervals = self.dbn.st.state_intervals[path]
        bins = np.bincount(intervals, minlength=self.dbn.st.intervals.max() + 1)
        bins = bins[self.dbn.st.intervals.min():]
        return (bins, self.dbn.st.intervals)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Compute the histogram of the beat intervals with a DBN using the
        forward algorithm.

        Parameters
        ----------
        activations : numpy float
            Beat activation function.
        reset : bool, optional
            Reset DBN to initial state before processing.

        Returns
        -------
        histogram_bins : numpy array
           Bins of the tempo histogram.
        histogram_delays : numpy array
           Corresponding delays [frames].

        """
        if reset:
            self.reset()
        fwd = self.dbn.hmm.forward(activations, reset=reset)
        states = np.argmax(fwd, axis=1)
        intervals = self.dbn.st.state_intervals[states]
        bins = np.zeros((len(activations), len(self.intervals)))
        bins[np.arange(len(activations)), intervals - self.min_interval] = 1
        bins = self._hist_buffer(bins)
        return (np.sum(bins, axis=0), self.intervals)