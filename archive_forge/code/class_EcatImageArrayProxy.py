import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
class EcatImageArrayProxy:
    """Ecat implementation of array proxy protocol

    The array proxy allows us to freeze the passed fileobj and
    header such that it returns the expected data array.
    """

    def __init__(self, subheader):
        self._subheader = subheader
        self._data = None
        x, y, z = subheader.get_shape()
        nframes = subheader.get_nframes()
        self._shape = (x, y, z, nframes)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_proxy(self):
        return True

    def __array__(self, dtype=None):
        """Read of data from file

        This reads ALL FRAMES into one array, can be memory expensive.

        If you want to read only some slices, use the slicing syntax
        (``__getitem__``) below, or ``subheader.data_from_fileobj(frame)``

        Parameters
        ----------
        dtype : numpy dtype specifier, optional
            A numpy dtype specifier specifying the type of the returned array.

        Returns
        -------
        array
            Scaled image data with type `dtype`.
        """
        data = np.empty(self.shape)
        frame_mapping = get_frame_order(self._subheader._mlist)
        for i in sorted(frame_mapping):
            data[:, :, :, i] = self._subheader.data_from_fileobj(frame_mapping[i][0])
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        return data

    def __getitem__(self, sliceobj):
        """Return slice `sliceobj` from ECAT data, optimizing if possible"""
        sliceobj = canonical_slicers(sliceobj, self.shape)
        ax_inds = [i for i, obj in enumerate(sliceobj) if obj is not None]
        assert len(ax_inds) == len(self.shape)
        frame_mapping = get_frame_order(self._subheader._mlist)
        slice3 = sliceobj[ax_inds[3]]
        in_slicer = sliceobj[:ax_inds[3]] + sliceobj[ax_inds[3] + 1:]
        if isinstance(slice3, Integral):
            data = self._subheader.data_from_fileobj(frame_mapping[slice3][0])
            return data[in_slicer]
        out_shape = predict_shape(sliceobj, self.shape)
        out_data = np.empty(out_shape)
        out_slicer = [slice(None)] * len(out_shape)
        in2out_ind = slice2outax(len(self.shape), sliceobj)[3]
        for i in list(range(self.shape[3]))[slice3]:
            data = self._subheader.data_from_fileobj(frame_mapping[i][0])
            out_slicer[in2out_ind] = i
            out_data[tuple(out_slicer)] = data[in_slicer]
        return out_data