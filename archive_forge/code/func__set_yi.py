import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def _set_yi(self, yi, xi=None, axis=None):
    if axis is None:
        axis = self._y_axis
    if axis is None:
        raise ValueError('no interpolation axis specified')
    shape = yi.shape
    if shape == ():
        shape = (1,)
    if xi is not None and shape[axis] != len(xi):
        raise ValueError('x and y arrays must be equal in length along interpolation axis.')
    self._y_axis = axis % yi.ndim
    self._y_extra_shape = yi.shape[:self._y_axis] + yi.shape[self._y_axis + 1:]
    self.dtype = None
    self._set_dtype(yi.dtype)