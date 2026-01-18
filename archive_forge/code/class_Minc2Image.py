import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
class Minc2Image(Minc1Image):
    """Class for MINC2 images

    The MINC2 image class uses the default header type, rather than a
    specific MINC header type - and reads the relevant information from
    the MINC file on load.
    """
    _compressed_suffixes = ()
    header_class = Minc2Header
    header: Minc2Header

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        import h5py
        holder = file_map['image']
        if holder.filename is None:
            raise MincError('MINC2 needs filename for load')
        minc_file = Minc2File(h5py.File(holder.filename, 'r'))
        affine = minc_file.get_affine()
        if affine.shape != (4, 4):
            raise MincError('Image does not have 3 spatial dimensions')
        data_dtype = minc_file.get_data_dtype()
        shape = minc_file.get_data_shape()
        zooms = minc_file.get_zooms()
        header = klass.header_class(data_dtype, shape, zooms)
        data = klass.ImageArrayProxy(minc_file)
        return klass(data, affine, header, extra=None, file_map=file_map)