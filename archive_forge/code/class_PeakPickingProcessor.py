from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
class PeakPickingProcessor(Processor):
    """
    Deprecated as of version 0.15. Will be removed in version 0.16. Use either
    :class:`OnsetPeakPickingProcessor` or :class:`NotePeakPickingProcessor`
    instead.

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def process(self, activations, **kwargs):
        """
        Detect the peaks in the given activation function.

        Parameters
        ----------
        activations : numpy array
            Onset activation function.

        Returns
        -------
        peaks : numpy array
            Detected onsets [seconds[, frequency bin]].

        """
        import warnings
        if activations.ndim == 1:
            warnings.warn('`PeakPickingProcessor` is deprecated as of version 0.15 and will be removed in version 0.16. Use `OnsetPeakPickingProcessor` instead.')
            ppp = OnsetPeakPickingProcessor(**self.kwargs)
            return ppp(activations, **kwargs)
        elif activations.ndim == 2:
            warnings.warn('`PeakPickingProcessor` is deprecated as of version 0.15 and will be removed in version 0.16. Use `NotePeakPickingProcessor` instead.')
            from .notes import NotePeakPickingProcessor
            ppp = NotePeakPickingProcessor(**self.kwargs)
            return ppp(activations, **kwargs)

    @staticmethod
    def add_arguments(parser, **kwargs):
        """
        Deprecated as of version 0.15. Will be removed in version 0.16. Use
        either :class:`OnsetPeakPickingProcessor` or
        :class:`NotePeakPickingProcessor` instead.

        """
        return OnsetPeakPickingProcessor.add_arguments(parser, **kwargs)