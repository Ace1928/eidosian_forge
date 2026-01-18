from __future__ import annotations
import io
import typing as ty
from collections.abc import Sequence
from typing import Literal
import numpy as np
from .arrayproxy import ArrayLike
from .casting import sctypes_aliases
from .dataobj_images import DataobjImage
from .filebasedimages import FileBasedHeader, FileBasedImage
from .fileholders import FileMap
from .fileslice import canonical_slicers
from .orientations import apply_orientation, inv_ornt_aff
from .viewers import OrthoSlicer3D
from .volumeutils import shape_zoom_affine
class SpatialHeader(FileBasedHeader, SpatialProtocol):
    """Template class to implement header protocol"""
    default_x_flip: bool = True
    data_layout: Literal['F', 'C'] = 'F'
    _dtype: np.dtype
    _shape: tuple[int, ...]
    _zooms: tuple[float, ...]

    def __init__(self, data_dtype: npt.DTypeLike=np.float32, shape: Sequence[int]=(0,), zooms: Sequence[float] | None=None):
        self.set_data_dtype(data_dtype)
        self._zooms = ()
        self.set_data_shape(shape)
        if zooms is not None:
            self.set_zooms(zooms)

    @classmethod
    def from_header(klass: type[SpatialHdrT], header: SpatialProtocol | FileBasedHeader | ty.Mapping | None=None) -> SpatialHdrT:
        if header is None:
            return klass()
        if type(header) == klass:
            return header.copy()
        if isinstance(header, SpatialProtocol):
            return klass(header.get_data_dtype(), header.get_data_shape(), header.get_zooms())
        return super().from_header(header)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SpatialHeader):
            return (self.get_data_dtype(), self.get_data_shape(), self.get_zooms()) == (other.get_data_dtype(), other.get_data_shape(), other.get_zooms())
        return NotImplemented

    def copy(self: SpatialHdrT) -> SpatialHdrT:
        """Copy object to independent representation

        The copy should not be affected by any changes to the original
        object.
        """
        return self.__class__(self._dtype, self._shape, self._zooms)

    def get_data_dtype(self) -> np.dtype:
        return self._dtype

    def set_data_dtype(self, dtype: npt.DTypeLike) -> None:
        self._dtype = np.dtype(dtype)

    def get_data_shape(self) -> tuple[int, ...]:
        return self._shape

    def set_data_shape(self, shape: Sequence[int]) -> None:
        ndim = len(shape)
        if ndim == 0:
            self._shape = (0,)
            self._zooms = (1.0,)
            return
        self._shape = tuple((int(s) for s in shape))
        nzs = min(len(self._zooms), ndim)
        self._zooms = self._zooms[:nzs] + (1.0,) * (ndim - nzs)

    def get_zooms(self) -> tuple[float, ...]:
        return self._zooms

    def set_zooms(self, zooms: Sequence[float]) -> None:
        zooms = tuple((float(z) for z in zooms))
        shape = self.get_data_shape()
        ndim = len(shape)
        if len(zooms) != ndim:
            raise HeaderDataError('Expecting %d zoom values for ndim %d' % (ndim, ndim))
        if any((z < 0 for z in zooms)):
            raise HeaderDataError('zooms must be positive')
        self._zooms = zooms

    def get_base_affine(self) -> np.ndarray:
        shape = self.get_data_shape()
        zooms = self.get_zooms()
        return shape_zoom_affine(shape, zooms, self.default_x_flip)
    get_best_affine = get_base_affine

    def data_to_fileobj(self, data: npt.ArrayLike, fileobj: io.IOBase, rescale: bool=True):
        """Write array data `data` as binary to `fileobj`

        Parameters
        ----------
        data : array-like
            data to write
        fileobj : file-like object
            file-like object implementing 'write'
        rescale : {True, False}, optional
            Whether to try and rescale data to match output dtype specified by
            header. For this minimal header, `rescale` has no effect
        """
        data = np.asarray(data)
        dtype = self.get_data_dtype()
        fileobj.write(data.astype(dtype).tobytes(order=self.data_layout))

    def data_from_fileobj(self, fileobj: io.IOBase) -> np.ndarray:
        """Read binary image data from `fileobj`"""
        dtype = self.get_data_dtype()
        shape = self.get_data_shape()
        data_size = int(np.prod(shape) * dtype.itemsize)
        data_bytes = fileobj.read(data_size)
        return np.ndarray(shape, dtype, data_bytes, order=self.data_layout)