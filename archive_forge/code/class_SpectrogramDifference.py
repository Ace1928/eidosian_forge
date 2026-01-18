from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
class SpectrogramDifference(Spectrogram):
    """
    SpectrogramDifference class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
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
    keep_dims : bool, optional
        Indicate if the dimensions (i.e. shape) of the spectrogram should be
        kept.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    The first `diff_frames` frames will have a value of 0.

    If `keep_dims` is 'True' the returned difference has the same shape as the
    spectrogram. This is needed if the diffs should be stacked on top of it.
    If set to 'False', the length will be `diff_frames` frames shorter (mostly
    used by the SpectrogramDifferenceProcessor which first buffers that many
    frames.

    The SuperFlux algorithm [1]_ uses a maximum filtered spectrogram with 3
    `diff_max_bins` together with a 24 band logarithmic filterbank to calculate
    the difference spectrogram with a `diff_ratio` of 0.5.

    The effect of this maximum filter applied to the spectrogram is that the
    magnitudes are "widened" in frequency direction, i.e. the following
    difference calculation is less sensitive against frequency fluctuations.
    This effect is exploited to suppress false positive energy fragments
    originating from vibrato.

    References
    ----------
    .. [1] Sebastian Böck and Gerhard Widmer
           "Maximum Filter Vibrato Suppression for Onset Detection"
           Proceedings of the 16th International Conference on Digital Audio
           Effects (DAFx), 2013.

    Examples
    --------
    To obtain the SuperFlux feature as described above first create a filtered
    and logarithmically spaced spectrogram:

    >>> spec = LogarithmicFilteredSpectrogram('tests/data/audio/sample.wav',                                               num_bands=24, fps=200)
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicFilteredSpectrogram([[0.82358, 0.86341, ..., 0.02809, 0.02672],
                                    [0.92514, 0.93211, ..., 0.03607, 0.0317 ],
                                    ...,
                                    [1.03826, 0.767  , ..., 0.01814, 0.01138],
                                    [0.98236, 0.89276, ..., 0.01669, 0.00919]],
                                    dtype=float32)
    >>> spec.shape
    (561, 140)

    Then use the temporal first order difference and apply a maximum filter
    with 3 bands, keeping only the positive differences (i.e. rise in energy):

    >>> superflux = SpectrogramDifference(spec, diff_max_bins=3,                                           positive_diffs=True)
    >>> superflux  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    SpectrogramDifference([[0.     , 0. , ...,  0. ,  0. ],
                           [0.     , 0. , ...,  0. ,  0. ],
                           ...,
                           [0.01941, 0. , ...,  0. ,  0. ],
                           [0.     , 0. , ...,  0. ,  0. ]], dtype=float32)

    """

    def __init__(self, spectrogram, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS, positive_diffs=POSITIVE_DIFFS, keep_dims=True, **kwargs):
        pass

    def __new__(cls, spectrogram, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS, positive_diffs=POSITIVE_DIFFS, keep_dims=True, **kwargs):
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        if diff_frames is None:
            diff_frames = _diff_frames(diff_ratio, hop_size=spectrogram.stft.frames.hop_size, frame_size=spectrogram.stft.frames.frame_size, window=spectrogram.stft.window)
        if diff_max_bins is not None and diff_max_bins > 1:
            from scipy.ndimage.filters import maximum_filter
            size = (1, int(diff_max_bins))
            diff_spec = maximum_filter(spectrogram, size=size)
        else:
            diff_spec = spectrogram
        if keep_dims:
            diff = np.zeros_like(spectrogram)
            diff[diff_frames:] = spectrogram[diff_frames:] - diff_spec[:-diff_frames]
        else:
            diff = spectrogram[diff_frames:] - diff_spec[:-diff_frames]
        if positive_diffs:
            np.maximum(diff, 0, out=diff)
        obj = np.asarray(diff).view(cls)
        obj.spectrogram = spectrogram
        obj.diff_ratio = diff_ratio
        obj.diff_frames = diff_frames
        obj.diff_max_bins = diff_max_bins
        obj.positive_diffs = positive_diffs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.diff_ratio = getattr(obj, 'diff_ratio', 0.5)
        self.diff_frames = getattr(obj, 'diff_frames', None)
        self.diff_max_bins = getattr(obj, 'diff_max_bins', None)
        self.positive_diffs = getattr(obj, 'positive_diffs', False)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.spectrogram.bin_frequencies

    def positive_diff(self):
        """Positive diff."""
        return np.maximum(self, 0)