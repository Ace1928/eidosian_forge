from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.audio.spectrogram import SemitoneBandpassSpectrogram
from madmom.processors import Processor, SequentialProcessor
class CLPChroma(np.ndarray):
    """
    Compressed Log Pitch (CLP) chroma as proposed in [1]_ and [2]_.

    Parameters
    ----------
    data : str, Signal, or SemitoneBandpassSpectrogram
        Input data.
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

    Notes
    -----
    The resulting chromagrams differ slightly from those obtained by the
    MATLAB chroma toolbox [2]_ because of different resampling and filter
    methods.

    References
    ----------
    .. [1] Meinard Müller,
           "Information retrieval for music and motion", Springer, 2007.

    .. [2] Meinard Müller and Sebastian Ewert,
           "Chroma Toolbox: MATLAB Implementations for Extracting Variants of
           Chroma-Based Audio Features",
           Proceedings of the International Conference on Music Information
           Retrieval (ISMIR), 2011.

    """

    def __init__(self, data, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX, compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM, threshold=CLP_THRESHOLD, **kwargs):
        pass

    def __new__(cls, data, fps=CLP_FPS, fmin=CLP_FMIN, fmax=CLP_FMAX, compression_factor=CLP_COMPRESSION_FACTOR, norm=CLP_NORM, threshold=CLP_THRESHOLD, **kwargs):
        from madmom.audio.filters import hz2midi
        if not isinstance(data, SemitoneBandpassSpectrogram):
            data = SemitoneBandpassSpectrogram(data, fps=fps, fmin=fmin, fmax=fmax)
        log_pitch_energy = np.log10(data * compression_factor + 1)
        obj = np.zeros((log_pitch_energy.shape[0], 12)).view(cls)
        midi_min = int(np.round(hz2midi(data.bin_frequencies[0])))
        for p in range(log_pitch_energy.shape[1]):
            chroma_idx = np.mod(midi_min + p, 12)
            obj[:, chroma_idx] += log_pitch_energy[:, p]
        obj.bin_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        obj.fps = fps
        if norm:
            mean_energy = np.sqrt((obj ** 2).sum(axis=1))
            idx_below_threshold = np.where(mean_energy < threshold)
            obj /= mean_energy[:, np.newaxis]
            obj[idx_below_threshold, :] = np.ones((1, 12)) / np.sqrt(12)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.fps = getattr(obj, 'fps', None)
        self.bin_labels = getattr(obj, 'bin_labels', None)