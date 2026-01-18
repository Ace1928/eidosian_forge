from __future__ import annotations
import numpy as np
from onnx.reference.op_run import OpRun
def _cfft(x: np.ndarray, fft_length: int, axis: int, onesided: bool, normalize: bool) -> np.ndarray:
    if x.shape[-1] == 1:
        signal = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        signal = real + 1j * imag
    complex_signals = np.squeeze(signal, -1)
    result = _fft(complex_signals, fft_length, axis=axis)
    if onesided:
        slices = [slice(0, a) for a in result.shape]
        slices[axis] = slice(0, result.shape[axis] // 2 + 1)
        result = result[tuple(slices)]
    if normalize:
        result /= fft_length
    return result