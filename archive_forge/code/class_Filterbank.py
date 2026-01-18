from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
class Filterbank(np.ndarray):
    """
    Generic filterbank class.

    A Filterbank is a simple numpy array enhanced with several additional
    attributes, e.g. number of bands.

    A Filterbank has a shape of (num_bins, num_bands) and can be used to
    filter a spectrogram of shape (num_frames, num_bins) to (num_frames,
    num_bands).

    Parameters
    ----------
    data : numpy array, shape (num_bins, num_bands)
        Data of the filterbank .
    bin_frequencies : numpy array, shape (num_bins, )
        Frequencies of the bins [Hz].

    Notes
    -----
    The length of `bin_frequencies` must be equal to the first dimension
    of the given `data` array.

    """

    def __init__(self, data, bin_frequencies):
        pass

    def __new__(cls, data, bin_frequencies):
        if isinstance(data, np.ndarray) and data.ndim == 2:
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError('wrong input data for Filterbank, must be a 2D np.ndarray')
        if len(bin_frequencies) != obj.shape[0]:
            raise ValueError('`bin_frequencies` must have the same length as the first dimension of `data`.')
        obj.bin_frequencies = np.asarray(bin_frequencies, dtype=np.float)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)

    @classmethod
    def _put_filter(cls, filt, band):
        """
        Puts a filter in the band, internal helper function.

        Parameters
        ----------
        filt : :class:`Filter` instance
            Filter to be put into the band.
        band : numpy array
            Band in which the filter should be put.

        Notes
        -----
        The `band` must be an existing numpy array where the filter `filt` is
        put in, given the position of the filter. Out of range filters are
        truncated. If there are non-zero values in the filter band at the
        respective positions, the maximum value of the `band` and the filter
        `filt` is used.

        """
        if not isinstance(filt, Filter):
            raise ValueError('unable to determine start position of Filter')
        start = filt.start
        stop = start + len(filt)
        if start < 0:
            filt = filt[-start:]
            start = 0
        if stop > len(band):
            filt = filt[:-(stop - len(band))]
            stop = len(band)
        filter_position = band[start:stop]
        np.maximum(filt, filter_position, out=filter_position)

    @classmethod
    def from_filters(cls, filters, bin_frequencies):
        """
        Create a filterbank with possibly multiple filters per band.

        Parameters
        ----------
        filters : list (of lists) of Filters
            List of Filters (per band); if multiple filters per band are
            desired, they should be also contained in a list, resulting in a
            list of lists of Filters.
        bin_frequencies : numpy array
            Frequencies of the bins (needed to determine the expected size of
            the filterbank).

        Returns
        -------
        filterbank : :class:`Filterbank` instance
            Filterbank with respective filter elements.

        """
        fb = np.zeros((len(bin_frequencies), len(filters)))
        for band_id, band_filter in enumerate(filters):
            band = fb[:, band_id]
            if isinstance(band_filter, list):
                for filt in band_filter:
                    cls._put_filter(filt, band)
            else:
                cls._put_filter(band_filter, band)
        return Filterbank.__new__(cls, fb, bin_frequencies)

    @property
    def num_bins(self):
        """Number of bins."""
        return self.shape[0]

    @property
    def num_bands(self):
        """Number of bands."""
        return self.shape[1]

    @property
    def corner_frequencies(self):
        """Corner frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            bins = np.nonzero(self[:, band])[0]
            freqs.append([np.min(bins), np.max(bins)])
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def center_frequencies(self):
        """Center frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            bins = np.nonzero(self[:, band])[0]
            min_bin = np.min(bins)
            max_bin = np.max(bins)
            if self[min_bin, band] == self[max_bin, band]:
                center = int(min_bin + (max_bin - min_bin) / 2.0)
            else:
                center = min_bin + np.argmax(self[min_bin:max_bin, band])
            freqs.append(center)
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][0]]

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][-1]]