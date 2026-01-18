import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def _interpolate_multikeys(self, x, y, tri_index=None, return_keys=('z',)):
    """
        Versatile (private) method defined for all TriInterpolators.

        :meth:`_interpolate_multikeys` is a wrapper around method
        :meth:`_interpolate_single_key` (to be defined in the child
        subclasses).
        :meth:`_interpolate_single_key actually performs the interpolation,
        but only for 1-dimensional inputs and at valid locations (inside
        unmasked triangles of the triangulation).

        The purpose of :meth:`_interpolate_multikeys` is to implement the
        following common tasks needed in all subclasses implementations:

        - calculation of containing triangles
        - dealing with more than one interpolation request at the same
          location (e.g., if the 2 derivatives are requested, it is
          unnecessary to compute the containing triangles twice)
        - scaling according to self._unit_x, self._unit_y
        - dealing with points outside of the grid (with fill value np.nan)
        - dealing with multi-dimensional *x*, *y* arrays: flattening for
          :meth:`_interpolate_params` call and final reshaping.

        (Note that np.vectorize could do most of those things very well for
        you, but it does it by function evaluations over successive tuples of
        the input arrays. Therefore, this tends to be more time-consuming than
        using optimized numpy functions - e.g., np.dot - which can be used
        easily on the flattened inputs, in the child-subclass methods
        :meth:`_interpolate_single_key`.)

        It is guaranteed that the calls to :meth:`_interpolate_single_key`
        will be done with flattened (1-d) array-like input parameters *x*, *y*
        and with flattened, valid `tri_index` arrays (no -1 index allowed).

        Parameters
        ----------
        x, y : array-like
            x and y coordinates where interpolated values are requested.
        tri_index : array-like of int, optional
            Array of the containing triangle indices, same shape as
            *x* and *y*. Defaults to None. If None, these indices
            will be computed by a TriFinder instance.
            (Note: For point outside the grid, tri_index[ipt] shall be -1).
        return_keys : tuple of keys from {'z', 'dzdx', 'dzdy'}
            Defines the interpolation arrays to return, and in which order.

        Returns
        -------
        list of arrays
            Each array-like contains the expected interpolated values in the
            order defined by *return_keys* parameter.
        """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sh_ret = x.shape
    if x.shape != y.shape:
        raise ValueError(f'x and y shall have same shapes. Given: {x.shape} and {y.shape}')
    x = np.ravel(x)
    y = np.ravel(y)
    x_scaled = x / self._unit_x
    y_scaled = y / self._unit_y
    size_ret = np.size(x_scaled)
    if tri_index is None:
        tri_index = self._trifinder(x, y)
    else:
        if tri_index.shape != sh_ret:
            raise ValueError(f'tri_index array is provided and shall have same shape as x and y. Given: {tri_index.shape} and {sh_ret}')
        tri_index = np.ravel(tri_index)
    mask_in = tri_index != -1
    if self._tri_renum is None:
        valid_tri_index = tri_index[mask_in]
    else:
        valid_tri_index = self._tri_renum[tri_index[mask_in]]
    valid_x = x_scaled[mask_in]
    valid_y = y_scaled[mask_in]
    ret = []
    for return_key in return_keys:
        try:
            return_index = {'z': 0, 'dzdx': 1, 'dzdy': 2}[return_key]
        except KeyError as err:
            raise ValueError("return_keys items shall take values in {'z', 'dzdx', 'dzdy'}") from err
        scale = [1.0, 1.0 / self._unit_x, 1.0 / self._unit_y][return_index]
        ret_loc = np.empty(size_ret, dtype=np.float64)
        ret_loc[~mask_in] = np.nan
        ret_loc[mask_in] = self._interpolate_single_key(return_key, valid_tri_index, valid_x, valid_y) * scale
        ret += [np.ma.masked_invalid(ret_loc.reshape(sh_ret), copy=False)]
    return ret