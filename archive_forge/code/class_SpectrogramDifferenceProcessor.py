from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class SpectrogramDifferenceProcessor(Processor):
    """
    Difference Spectrogram Processor class.

    Parameters
    ----------
    diff_ratio : float, optional
        Calculate the difference to the frame at which the window used for the
        STFT yields this ratio of the maximum height.
    diff_frames : int, optional
        Calculate the difference to the `diff_frames`-th previous frame (if
        set, this overrides the value calculated from the `diff_ratio`)
    diff_max_bins : int, optional
        Apply a maximum filter with this width (in bins in frequency dimension)
        to the spectrogram the difference is calculated to.
    positive_diffs : bool, optional
        Keep only the positive differences, i.e. set all diff values < 0 to 0.
    stack_diffs : numpy stacking function, optional
        If 'None', only the differences are returned. If set, the diffs are
        stacked with the underlying spectrogram data according to the `stack`
        function:

        - ``np.vstack``
          the differences and spectrogram are stacked vertically, i.e. in time
          direction,
        - ``np.hstack``
          the differences and spectrogram are stacked horizontally, i.e. in
          frequency direction,
        - ``np.dstack``
          the differences and spectrogram are stacked in depth, i.e. return
          them as a 3D representation with depth as the third dimension.

    """

    def __init__(self, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS, positive_diffs=POSITIVE_DIFFS, stack_diffs=None, **kwargs):
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs
        self.stack_diffs = stack_diffs
        self._buffer = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_buffer', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._buffer = None

    def process(self, data, reset=True, **kwargs):
        """
        Perform a temporal difference calculation on the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        reset : bool, optional
            Reset the spectrogram buffer before computing the difference.
        kwargs : dict
            Keyword arguments passed to :class:`SpectrogramDifference`.

        Returns
        -------
        diff : :class:`SpectrogramDifference` instance
            Spectrogram difference.

        Notes
        -----
        If `reset` is 'True', the first `diff_frames` differences will be 0.

        """
        args = dict(diff_ratio=self.diff_ratio, diff_frames=self.diff_frames, diff_max_bins=self.diff_max_bins, positive_diffs=self.positive_diffs)
        args.update(kwargs)
        if self.diff_frames is None:
            self.diff_frames = _diff_frames(args['diff_ratio'], frame_size=data.stft.frames.frame_size, hop_size=data.stft.frames.hop_size, window=data.stft.window)
        if self._buffer is None or reset:
            init = np.empty((self.diff_frames, data.shape[1]))
            init[:] = np.inf
            data = np.insert(data, 0, init, axis=0)
            self._buffer = BufferProcessor(init=data)
        else:
            data = self._buffer(data)
        diff = SpectrogramDifference(data, keep_dims=False, **args)
        diff[np.isinf(diff)] = 0
        if self.stack_diffs is None:
            return diff
        return self.stack_diffs((diff.spectrogram[self.diff_frames:], diff))

    def reset(self):
        """Reset the SpectrogramDifferenceProcessor."""
        self._buffer = None

    @staticmethod
    def add_arguments(parser, diff=None, diff_ratio=None, diff_frames=None, diff_max_bins=None, positive_diffs=None):
        """
        Add spectrogram difference related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        diff : bool, optional
            Take the difference of the spectrogram.
        diff_ratio : float, optional
            Calculate the difference to the frame at which the window used for
            the STFT yields this ratio of the maximum height.
        diff_frames : int, optional
            Calculate the difference to the `diff_frames`-th previous frame (if
            set, this overrides the value calculated from the `diff_ratio`)
        diff_max_bins : int, optional
            Apply a maximum filter with this width (in bins in frequency
            dimension) to the spectrogram the difference is calculated to.
        positive_diffs : bool, optional
            Keep only the positive differences, i.e. set all diff values < 0
            to 0.

        Returns
        -------
        argparse argument group
            Spectrogram difference argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        Only the `diff_frames` parameter behaves differently, it is included
        if either the `diff_ratio` is set or a value != 'None' is given.

        """
        g = parser.add_argument_group('spectrogram difference arguments')
        if diff is True:
            g.add_argument('--no_diff', dest='diff', action='store_false', help='use the spectrogram [default=differences of the spectrogram]')
        elif diff is False:
            g.add_argument('--diff', action='store_true', help='use the differences of the spectrogram [default=spectrogram]')
        if diff_ratio is not None:
            g.add_argument('--diff_ratio', action='store', type=float, default=diff_ratio, help='calculate the difference to the frame at which the window of the STFT have this ratio of the maximum height [default=%(default).1f]')
        if diff_ratio is not None or diff_frames:
            g.add_argument('--diff_frames', action='store', type=int, default=diff_frames, help='calculate the difference to the N-th previous frame (this overrides the value calculated with `diff_ratio`) [default=%(default)s]')
        if positive_diffs is True:
            g.add_argument('--all_diffs', dest='positive_diffs', action='store_false', help='keep both positive and negative diffs [default=only the positive diffs]')
        elif positive_diffs is False:
            g.add_argument('--positive_diffs', action='store_true', help='keep only positive diffs [default=positive and negative diffs]')
        if diff_max_bins is not None:
            g.add_argument('--max_bins', action='store', type=int, dest='diff_max_bins', default=diff_max_bins, help='apply a maximum filter with this width (in frequency bins) [default=%(default)d]')
        return g