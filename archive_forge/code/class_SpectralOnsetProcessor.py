from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
class SpectralOnsetProcessor(SequentialProcessor):
    """
    The SpectralOnsetProcessor class implements most of the common onset
    detection functions based on the magnitude or phase information of a
    spectrogram.

    Parameters
    ----------
    onset_method : str, optional
        Onset detection function. See `METHODS` for possible values.
    kwargs : dict, optional
        Keyword arguments passed to the pre-processing chain to obtain a
        spectral representation of the signal.

    Notes
    -----
    If the spectrogram should be filtered, the `filterbank` parameter must
    contain a valid Filterbank, if it should be scaled logarithmically, `log`
    must be set accordingly.

    References
    ----------
    .. [1] Paul Masri,
           "Computer Modeling of Sound for Transformation and Synthesis of
           Musical Signals",
           PhD thesis, University of Bristol, 1996.
    .. [2] Sebastian Böck and Gerhard Widmer,
           "Maximum Filter Vibrato Suppression for Onset Detection",
           Proceedings of the 16th International Conference on Digital Audio
           Effects (DAFx), 2013.

    Examples
    --------

    Create a SpectralOnsetProcessor and pass a file through the processor to
    obtain an onset detection function. Per default the spectral flux [1]_ is
    computed on a simple Spectrogram.

    >>> sodf = SpectralOnsetProcessor()
    >>> sodf  # doctest: +ELLIPSIS
    <madmom.features.onsets.SpectralOnsetProcessor object at 0x...>
    >>> sodf.processors[-1]  # doctest: +ELLIPSIS
    <function spectral_flux at 0x...>
    >>> sodf('tests/data/audio/sample.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([ 0. , 100.90121, ..., 26.30577, 20.94439], dtype=float32)

    The parameters passed to the signal pre-processing chain can be set when
    creating the SpectralOnsetProcessor. E.g. to obtain the SuperFlux [2]_
    onset detection function set these parameters:

    >>> from madmom.audio.filters import LogarithmicFilterbank
    >>> sodf = SpectralOnsetProcessor(onset_method='superflux', fps=200,
    ...                               filterbank=LogarithmicFilterbank,
    ...                               num_bands=24, log=np.log10)
    >>> sodf('tests/data/audio/sample.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([ 0. , 0. , 2.0868 , 1.02404, ..., 0.29888, 0.12122], dtype=float32)

    """
    METHODS = ['superflux', 'complex_flux', 'high_frequency_content', 'spectral_diff', 'spectral_flux', 'modified_kullback_leibler', 'phase_deviation', 'weighted_phase_deviation', 'normalized_weighted_phase_deviation', 'complex_domain', 'rectified_complex_domain']

    def __init__(self, onset_method='spectral_flux', **kwargs):
        import inspect
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import SpectrogramProcessor, FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
        if any((odf in onset_method for odf in ('phase', 'complex'))):
            kwargs['circular_shift'] = True
        kwargs['num_channels'] = 1
        sig = SignalProcessor(**kwargs)
        frames = FramedSignalProcessor(**kwargs)
        stft = ShortTimeFourierTransformProcessor(**kwargs)
        spec = SpectrogramProcessor(**kwargs)
        processors = [sig, frames, stft, spec]
        if 'filterbank' in kwargs.keys() and kwargs['filterbank'] is not None:
            processors.append(FilteredSpectrogramProcessor(**kwargs))
        if 'log' in kwargs.keys() and kwargs['log'] is not None:
            processors.append(LogarithmicSpectrogramProcessor(**kwargs))
        if not inspect.isfunction(onset_method):
            try:
                onset_method = globals()[onset_method]
            except KeyError:
                raise ValueError('%s not a valid onset detection function, choose %s.' % (onset_method, self.METHODS))
            processors.append(onset_method)
        super(SpectralOnsetProcessor, self).__init__(processors)

    @classmethod
    def add_arguments(cls, parser, onset_method=None):
        """
        Add spectral onset detection arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        onset_method : str, optional
            Default onset detection method.

        Returns
        -------
        parser_group : argparse argument group
            Spectral onset detection argument parser group.

        """
        g = parser.add_argument_group('spectral onset detection arguments')
        if onset_method is not None:
            g.add_argument('--odf', dest='onset_method', default=onset_method, choices=cls.METHODS, help='use this onset detection function [default=%(default)s]')
        return g