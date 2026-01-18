from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
class RNNOnsetProcessor(SequentialProcessor):
    """
    Processor to get a onset activation function from multiple RNNs.

    Parameters
    ----------
    online : bool, optional
        Choose networks suitable for online onset detection, i.e. use
        unidirectional RNNs.

    Notes
    -----
    This class uses either uni- or bi-directional RNNs. Contrary to [1], it
    uses simple tanh units as in [2]. Also the input representations changed
    to use logarithmically filtered and scaled spectrograms.

    References
    ----------
    .. [1] "Universal Onset Detection with bidirectional Long Short-Term Memory
           Neural Networks"
           Florian Eyben, Sebastian Böck, Björn Schuller and Alex Graves.
           Proceedings of the 11th International Society for Music Information
           Retrieval Conference (ISMIR), 2010.
    .. [2] "Online Real-time Onset Detection with Recurrent Neural Networks"
           Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl.
           Proceedings of the 15th International Conference on Digital Audio
           Effects (DAFx), 2012.

    Examples
    --------
    Create a RNNOnsetProcessor and pass a file through the processor to obtain
    an onset detection function (sampled with 100 frames per second).

    >>> proc = RNNOnsetProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.onsets.RNNOnsetProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav') # doctest: +ELLIPSIS
    array([0.08313, 0.0024 , ... 0.00527], dtype=float32)

    """

    def __init__(self, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor
        from ..models import ONSETS_RNN, ONSETS_BRNN
        from ..ml.nn import NeuralNetworkEnsemble
        if kwargs.get('online'):
            nn_files = ONSETS_RNN
            frame_sizes = [512, 1024, 2048]
        else:
            nn_files = ONSETS_BRNN
            frame_sizes = [1024, 2048, 4096]
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        for frame_size in frame_sizes:
            frames = FramedSignalProcessor(frame_size=frame_size, **kwargs)
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(num_bands=6, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=5, add=1)
            diff = SpectrogramDifferenceProcessor(diff_ratio=0.25, positive_diffs=True, stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        nn = NeuralNetworkEnsemble.load(nn_files, **kwargs)
        super(RNNOnsetProcessor, self).__init__((pre_processor, nn))