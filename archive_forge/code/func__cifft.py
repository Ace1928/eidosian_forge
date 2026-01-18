from __future__ import annotations
import numpy as np
from onnx.reference.op_run import OpRun
def _cifft(x: np.ndarray, fft_length: int, axis: int, onesided: bool=False) -> np.ndarray:
    if x.shape[-1] == 1:
        frequencies = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        frequencies = real + 1j * imag
    complex_frequencies = np.squeeze(frequencies, -1)
    return _ifft(complex_frequencies, fft_length, axis=axis, onesided=onesided)