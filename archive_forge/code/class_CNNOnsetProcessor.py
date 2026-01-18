from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
class CNNOnsetProcessor(SequentialProcessor):
    """
    Processor to get a onset activation function from a CNN.

    References
    ----------
    .. [1] "Musical Onset Detection with Convolutional Neural Networks"
           Jan Schlüter and Sebastian Böck.
           Proceedings of the 6th International Workshop on Machine Learning
           and Music, 2013.

    Notes
    -----
    The implementation follows as closely as possible the original one, but
    part of the signal pre-processing differs in minor aspects, so results can
    differ slightly, too.

    Examples
    --------
    Create a CNNOnsetProcessor and pass a file through the processor to obtain
    an onset detection function (sampled with 100 frames per second).

    >>> proc = CNNOnsetProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.onsets.CNNOnsetProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +ELLIPSIS
    array([0.05369, 0.04205, ... 0.00014], dtype=float32)

    """

    def __init__(self, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.filters import MelFilterbank
        from ..audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
        from ..models import ONSETS_CNN
        from ..ml.nn import NeuralNetwork
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        for frame_size in [2048, 1024, 4096]:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000, norm_filters=True, unique_filters=False)
            spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)
            multi.append(SequentialProcessor((frames, stft, filt, spec)))
        stack = np.dstack
        pad = _cnn_onset_processor_pad
        pre_processor = SequentialProcessor((sig, multi, stack, pad))
        nn = NeuralNetwork.load(ONSETS_CNN[0])
        super(CNNOnsetProcessor, self).__init__((pre_processor, nn))