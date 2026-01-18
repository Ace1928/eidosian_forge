import os
import re
from copy import deepcopy
import numpy as np
from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder
class AFNIArrayProxy(ArrayProxy):
    """Proxy object for AFNI image array.

    Attributes
    ----------
    scaling : np.ndarray
        Scaling factor (one factor per volume/sub-brick) for data. Default is
        None
    """

    def __init__(self, file_like, header, *, mmap=True, keep_file_open=None):
        """
        Initialize AFNI array proxy

        Parameters
        ----------
        file_like : file-like object
            File-like object or filename. If file-like object, should implement
            at least ``read`` and ``seek``.
        header : ``AFNIHeader`` object
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading data.
            If False, do not try numpy ``memmap`` for data array.  If one of
            {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
            True gives the same behavior as ``mmap='c'``.  If `file_like`
            cannot be memory-mapped, ignore `mmap` value and read array from
            file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_like`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.
        """
        super().__init__(file_like, header, mmap=mmap, keep_file_open=keep_file_open)
        self._scaling = header.get_data_scaling()

    @property
    def scaling(self):
        return self._scaling

    def _get_scaled(self, dtype, slicer):
        raw_data = self._get_unscaled(slicer=slicer)
        if self.scaling is None:
            if dtype is None:
                return raw_data
            final_type = np.promote_types(raw_data.dtype, dtype)
            return raw_data.astype(final_type, copy=False)
        fake_data = strided_scalar(self._shape)
        _, scaling = np.broadcast_arrays(fake_data, self.scaling)
        final_type = np.result_type(raw_data, scaling)
        if dtype is not None:
            final_type = np.promote_types(final_type, dtype)
        return raw_data * scaling[slicer].astype(final_type)