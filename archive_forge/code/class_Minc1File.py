from __future__ import annotations
from numbers import Integral
import numpy as np
from .externals.netcdf import netcdf_file
from .fileslice import canonical_slicers
from .spatialimages import SpatialHeader, SpatialImage
class Minc1File:
    """Class to wrap MINC1 format opened netcdf object

    Although it has some of the same methods as a ``Header``, we use
    this only when reading a MINC file, to pull out useful header
    information, and for the method of reading the data out
    """

    def __init__(self, mincfile):
        self._mincfile = mincfile
        self._image = mincfile.variables['image']
        self._dim_names = self._image.dimensions
        self._dims = [self._mincfile.variables[s] for s in self._dim_names]
        for dim in self._dims:
            if dim.spacing != b'regular__':
                raise ValueError('Irregular spacing not supported')
        self._spatial_dims = [name for name in self._dim_names if name.endswith('space')]
        self._image_max = self._mincfile.variables['image-max']
        self._image_min = self._mincfile.variables['image-min']

    def _get_dimensions(self, var):
        return var.dimensions

    def get_data_dtype(self):
        typecode = self._image.typecode()
        if typecode == 'f':
            dtt = np.dtype(np.float32)
        elif typecode == 'd':
            dtt = np.dtype(np.float64)
        else:
            signtype = self._image.signtype.decode('latin-1')
            dtt = _dt_dict[typecode, signtype]
        return np.dtype(dtt).newbyteorder('>')

    def get_data_shape(self):
        return self._image.data.shape

    def get_zooms(self):
        """Get real-world sizes of voxels"""
        return tuple((abs(float(dim.step)) if hasattr(dim, 'step') else 1.0 for dim in self._dims))

    def get_affine(self):
        nspatial = len(self._spatial_dims)
        rot_mat = np.eye(nspatial)
        steps = np.zeros((nspatial,))
        starts = np.zeros((nspatial,))
        dim_names = list(self._dim_names)
        for i, name in enumerate(self._spatial_dims):
            dim = self._dims[dim_names.index(name)]
            rot_mat[:, i] = dim.direction_cosines if hasattr(dim, 'direction_cosines') else _default_dir_cos[name]
            steps[i] = dim.step if hasattr(dim, 'step') else 1.0
            starts[i] = dim.start if hasattr(dim, 'start') else 0.0
        origin = np.dot(rot_mat, starts)
        aff = np.eye(nspatial + 1)
        aff[:nspatial, :nspatial] = rot_mat * steps
        aff[:nspatial, nspatial] = origin
        return aff

    def _get_valid_range(self):
        """Return valid range for image data

        The valid range can come from the image 'valid_range' or
        image 'valid_min' and 'valid_max', or, failing that, from the
        data type range
        """
        ddt = self.get_data_dtype()
        info = np.iinfo(ddt.type)
        try:
            valid_range = self._image.valid_range
        except AttributeError:
            try:
                valid_range = [self._image.valid_min, self._image.valid_max]
            except AttributeError:
                valid_range = [info.min, info.max]
        if valid_range[0] < info.min or valid_range[1] > info.max:
            raise ValueError('Valid range outside input data type range')
        return np.asarray(valid_range, dtype=np.float64)

    def _get_scalar(self, var):
        """Get scalar value from NetCDF scalar"""
        return var.getValue()

    def _get_array(self, var):
        """Get array from NetCDF array"""
        return var.data

    def _normalize(self, data, sliceobj=()):
        """Apply scaling to image data `data` already sliced with `sliceobj`

        https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide#Pixel_values_and_real_values

        MINC normalization uses "image-min" and "image-max" variables to
        map the data from the valid range of the image to the range
        specified by "image-min" and "image-max".

        The "image-max" and "image-min" are variables that describe the
        "max" and "min" of image over some dimensions of "image".

        The usual case is that "image" has dimensions ["zspace", "yspace",
        "xspace"] and "image-max" has dimensions ["zspace"], but there can be
        up to two dimensions for over which scaling is specified.

        Parameters
        ----------
        data : ndarray
            data after applying `sliceobj` slicing to full image
        sliceobj : tuple, optional
            slice definition. If not specified, assume no slicing has been
            applied to `data`
        """
        ddt = self.get_data_dtype()
        if np.issubdtype(ddt.type, np.floating):
            return data
        image_max = self._image_max
        image_min = self._image_min
        mx_dims = self._get_dimensions(image_max)
        mn_dims = self._get_dimensions(image_min)
        if mx_dims != mn_dims:
            raise MincError('"image-max" and "image-min" do not have the same dimensions')
        nscales = len(mx_dims)
        if nscales > 2:
            raise MincError('More than two scaling dimensions')
        if mx_dims != self._dim_names[:nscales]:
            raise MincError('image-max and image dimensions do not match')
        dmin, dmax = self._get_valid_range()
        out_data = np.clip(data, dmin, dmax)
        if nscales == 0:
            imax = self._get_scalar(image_max)
            imin = self._get_scalar(image_min)
        else:
            shape = self.get_data_shape()
            sliceobj = canonical_slicers(sliceobj, shape)
            ax_inds = [i for i, obj in enumerate(sliceobj) if obj is not None]
            assert len(ax_inds) == len(shape)
            nscales_ax = ax_inds[nscales]
            i_slicer = sliceobj[:nscales_ax]
            broad_part = tuple((None for s in sliceobj[ax_inds[nscales]:] if not isinstance(s, Integral)))
            i_slicer += broad_part
            imax = self._get_array(image_max)[i_slicer]
            imin = self._get_array(image_min)[i_slicer]
        slope = (imax - imin) / (dmax - dmin)
        inter = imin - dmin * slope
        out_data *= slope
        out_data += inter
        return out_data

    def get_scaled_data(self, sliceobj=()):
        """Return scaled data for slice definition `sliceobj`

        Parameters
        ----------
        sliceobj : tuple, optional
            slice definition. If not specified, return whole array

        Returns
        -------
        scaled_arr : array
            array from minc file with scaling applied
        """
        if sliceobj == ():
            raw_data = self._image.data
        else:
            raw_data = self._image.data[sliceobj]
        dtype = self.get_data_dtype()
        data = np.asarray(raw_data).view(dtype)
        return self._normalize(data, sliceobj)