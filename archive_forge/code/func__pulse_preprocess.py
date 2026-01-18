import cupy
from cupyx.scipy.signal import windows
def _pulse_preprocess(x, normalize, window):
    if window is not None:
        n = x.shape[-1]
        if callable(window):
            w = window(cupy.fft.fftfreq(n).astype(x.dtype))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (n,):
                raise ValueError('window must have the same length as data')
            w = window
        else:
            w = windows.get_window(window, n, False).astype(x.dtype)
        x = x * w
    if normalize:
        x = x / cupy.linalg.norm(x)
    return x