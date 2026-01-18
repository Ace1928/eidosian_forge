from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
class RNNBeatProcessor(SequentialProcessor):
    """
    Processor to get a beat activation function from multiple RNNs.

    Parameters
    ----------
    post_processor : Processor, optional
        Post-processor, default is to average the predictions.
    online : bool, optional
        Use signal processing parameters and RNN models suitable for online
        mode.
    nn_files : list, optional
        List with trained RNN model files. Per default ('None'), an ensemble
        of networks will be used.

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.

    Examples
    --------
    Create a RNNBeatProcessor and pass a file through the processor.
    The returned 1d array represents the probability of a beat at each frame,
    sampled at 100 frames per second.

    >>> proc = RNNBeatProcessor()
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.RNNBeatProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +ELLIPSIS
    array([0.00479, 0.00603, 0.00927, 0.01419, ... 0.02725], dtype=float32)

    For online processing, `online` must be set to 'True'. If processing power
    is limited, fewer number of RNN models can be defined via `nn_files`. The
    audio signal is then processed frame by frame.

    >>> from madmom.models import BEATS_LSTM
    >>> proc = RNNBeatProcessor(online=True, nn_files=[BEATS_LSTM[0]])
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.RNNBeatProcessor object at 0x...>
    >>> proc('tests/data/audio/sample.wav')  # doctest: +ELLIPSIS
    array([0.03887, 0.02619, 0.00747, 0.00218, ... 0.04825], dtype=float32)

    """

    def __init__(self, post_processor=average_predictions, online=False, nn_files=None, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import BEATS_LSTM, BEATS_BLSTM
        if online:
            if nn_files is None:
                nn_files = BEATS_LSTM
            frame_sizes = [2048]
            num_bands = 12
        else:
            if nn_files is None:
                nn_files = BEATS_BLSTM
            frame_sizes = [1024, 2048, 4096]
            num_bands = 6
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        multi = ParallelProcessor([])
        for frame_size in frame_sizes:
            frames = FramedSignalProcessor(frame_size=frame_size, **kwargs)
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        pre_processor = SequentialProcessor((sig, multi, np.hstack))
        nn = NeuralNetworkEnsemble.load(nn_files, ensemble_fn=post_processor, **kwargs)
        super(RNNBeatProcessor, self).__init__((pre_processor, nn))