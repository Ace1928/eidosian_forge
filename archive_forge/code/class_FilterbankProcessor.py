from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
class FilterbankProcessor(Processor, Filterbank):
    """
    Generic filterbank processor class.

    A FilterbankProcessor is a simple wrapper for Filterbank which adds a
    process() method.

    See Also
    --------
    :class:`Filterbank`

    """

    def process(self, data):
        """
        Filter the given data with the Filterbank.

        Parameters
        ----------
        data : 2D numpy array
            Data to be filtered.
        Returns
        -------
        filt_data : numpy array
            Filtered data.

        Notes
        -----
        This method makes the :class:`Filterbank` act as a :class:`Processor`.

        """
        return np.dot(data, self)

    @staticmethod
    def add_arguments(parser, filterbank=None, num_bands=None, crossover_frequencies=None, fmin=None, fmax=None, norm_filters=None, unique_filters=None):
        """
        Add filterbank related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        filterbank : :class:`.audio.filters.Filterbank`, optional
            Use a filterbank of that type.
        num_bands : int or list, optional
            Number of bands (per octave).
        crossover_frequencies : list or numpy array, optional
            List of crossover frequencies at which the `spectrogram` is split
            into bands.
        fmin : float, optional
            Minimum frequency of the filterbank [Hz].
        fmax : float, optional
            Maximum frequency of the filterbank [Hz].
        norm_filters : bool, optional
            Normalize the filters of the filterbank to area 1.
        unique_filters : bool, optional
            Indicate if the filterbank should contain only unique filters,
            i.e. remove duplicate filters resulting from insufficient
            resolution at low frequencies.

        Returns
        -------
        argparse argument group
            Filterbank argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.
        Depending on the type of the `filterbank`, either `num_bands` or
        `crossover_frequencies` should be used.

        """
        from madmom.utils import OverrideDefaultListAction
        g = parser.add_argument_group('filterbank arguments')
        if filterbank is not None:
            if issubclass(filterbank, Filterbank):
                g.add_argument('--no_filter', dest='filterbank', action='store_false', default=filterbank, help='do not filter the spectrogram with a filterbank [default=%(default)s]')
            else:
                g.add_argument('--filterbank', action='store_true', default=None, help='filter the spectrogram with a filterbank of this type')
        if isinstance(num_bands, int):
            g.add_argument('--num_bands', action='store', type=int, default=num_bands, help='number of filter bands (per octave) [default=%(default)i]')
        elif isinstance(num_bands, list):
            g.add_argument('--num_bands', type=int, default=num_bands, action=OverrideDefaultListAction, sep=',', help='(comma separated list of) number of filter bands (per octave) [default=%(default)s]')
        if crossover_frequencies is not None:
            g.add_argument('--crossover_frequencies', type=float, sep=',', action=OverrideDefaultListAction, default=crossover_frequencies, help='(comma separated) list with crossover frequencies [Hz, default=%(default)s]')
        if fmin is not None:
            g.add_argument('--fmin', action='store', type=float, default=fmin, help='minimum frequency of the filterbank [Hz, default=%(default).1f]')
        if fmax is not None:
            g.add_argument('--fmax', action='store', type=float, default=fmax, help='maximum frequency of the filterbank [Hz, default=%(default).1f]')
        if norm_filters is True:
            g.add_argument('--no_norm_filters', dest='norm_filters', action='store_false', default=norm_filters, help='do not normalize the filters to area 1 [default=True]')
        elif norm_filters is False:
            g.add_argument('--norm_filters', dest='norm_filters', action='store_true', default=norm_filters, help='normalize the filters to area 1 [default=False]')
        if unique_filters is True:
            g.add_argument('--duplicate_filters', dest='unique_filters', action='store_false', default=unique_filters, help='keep duplicate filters resulting from insufficient resolution at low frequencies [default=only unique filters are kept]')
        elif unique_filters is False:
            g.add_argument('--unique_filters', action='store_true', default=unique_filters, help='keep only unique filters, i.e. remove duplicate filters resulting from insufficient resolution at low frequencies [default=duplicate filters are kept]')
        return g