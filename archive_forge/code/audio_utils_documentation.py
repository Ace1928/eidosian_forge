import warnings
from typing import Optional, Union
import numpy as np

    Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same results
    as `torch.stft`.

    Args:
        frames (`np.array` of dimension `(num_frames, fft_window_size)`):
            A framed audio signal obtained using `audio_utils.fram_wav`.
        windowing_function (`np.array` of dimension `(nb_frequency_bins, nb_mel_filters)`:
            A array reprensenting the function that will be used to reduces the amplitude of the discontinuities at the
            boundaries of each frame when computing the STFT. Each frame will be multiplied by the windowing_function.
            For more information on the discontinuities, called *Spectral leakage*, refer to [this
            tutorial]https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
        fft_window_size (`int`, *optional*):
            Size of the window om which the Fourier transform is applied. This controls the frequency resolution of the
            spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples. The number of
            frequency bins (`nb_frequency_bins`) used to divide the window into equal strips is equal to
            `(1+fft_window_size)//2`. An increase of the fft_window_size slows the calculus time proportionnally.

    Example:

    ```python
    >>> from transformers.audio_utils import stft, fram_wave
    >>> import numpy as np

    >>> audio = np.random.rand(50)
    >>> fft_window_size = 10
    >>> hop_length = 2
    >>> framed_audio = fram_wave(audio, hop_length, fft_window_size)
    >>> spectrogram = stft(framed_audio, np.hanning(fft_window_size + 1))
    ```

    Returns:
        spectrogram (`np.ndarray`):
            A spectrogram of shape `(num_frames, nb_frequency_bins)` obtained using the STFT algorithm
    