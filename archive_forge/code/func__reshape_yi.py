import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def _reshape_yi(self, yi, check=False):
    """
        Reshape the updated yi to a 1-D array
        """
    yi = cupy.moveaxis(yi, self._y_axis, 0)
    if check and yi.shape[1:] != self._y_extra_shape:
        ok_shape = '%r + (N,) + %r' % (self._y_extra_shape[-self._y_axis:], self._y_extra_shape[:-self._y_axis])
        raise ValueError('Data must be of shape %s' % ok_shape)
    return yi.reshape((yi.shape[0], -1))