from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.audio.spectrogram import SemitoneBandpassSpectrogram
from madmom.processors import Processor, SequentialProcessor
class CLPChromaProcessor(Processor):
    """
    Compressed Log Pitch (CLP) Chroma Processor.

    Parameters
    ----------
    fps : int, optional
        Desired frame rate of the signal [Hz].
    fmin : float, optional
        Lowest frequency of the spectrogram [Hz].
    fmax : float, optional
        Highest frequency of the spectrogram [Hz].
    compression_factor : float, optional
        Factor for compression of the energy.
    norm : bool, optional
        Normalize the energy of each frame to one (divide by the L2 norm).
    threshold : float, optional
        If the energy of a frame is below a threshold, the energy is equally
        distributed among all chroma bins.

    """

    def __init__(self, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX, compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM, threshold=CLP_THRESHOLD, **kwargs):
        self.fps = fps
        self.fmin = fmin
        self.fmax = fmax
        self.compression_factor = compression_factor
        self.norm = norm
        self.threshold = threshold

    def process(self, data, **kwargs):
        """
        Create a CLPChroma from the given data.

        Parameters
        ----------
        data : Signal instance or filename
            Data to be processed.

        Returns
        -------
        clp : :class:`CLPChroma` instance
            CLPChroma.

        """
        args = dict(fps=self.fps, fmin=self.fmin, fmax=self.fmax, compression_factor=self.compression_factor, norm=self.norm, threshold=self.threshold)
        args.update(kwargs)
        return CLPChroma(data, **args)