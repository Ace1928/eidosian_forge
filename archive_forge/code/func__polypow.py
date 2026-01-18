import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
def _polypow(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    method = cupy._math.misc._choose_conv_method(x, x, 'full')
    if method == 'direct':
        return _polypow_direct(x, n)
    elif method == 'fft':
        if x.dtype.kind == 'c':
            fft, ifft = (cupy.fft.fft, cupy.fft.ifft)
        else:
            fft, ifft = (cupy.fft.rfft, cupy.fft.irfft)
        out_size = (x.size - 1) * n + 1
        size = cupyx.scipy.fft.next_fast_len(out_size)
        fx = fft(x, size)
        fy = cupy.power(fx, n, fx)
        y = ifft(fy, size)
        return y[:out_size]
    else:
        assert False