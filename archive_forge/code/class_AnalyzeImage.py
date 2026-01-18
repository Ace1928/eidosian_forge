from __future__ import annotations
import numpy as np
from .arrayproxy import ArrayProxy
from .arraywriters import ArrayWriter, WriterError, get_slope_inter, make_array_writer
from .batteryrunners import Report
from .fileholders import copy_file_map
from .spatialimages import HeaderDataError, HeaderTypeError, SpatialHeader, SpatialImage
from .volumeutils import (
from .wrapstruct import LabeledWrapStruct
class AnalyzeImage(SpatialImage):
    """Class for basic Analyze format image"""
    header_class: type[AnalyzeHeader] = AnalyzeHeader
    header: AnalyzeHeader
    _meta_sniff_len = header_class.sizeof_hdr
    files_types: tuple[tuple[str, str], ...] = (('image', '.img'), ('header', '.hdr'))
    valid_exts: tuple[str, ...] = ('.img', '.hdr')
    _compressed_suffixes: tuple[str, ...] = ('.gz', '.bz2', '.zst')
    makeable = True
    rw = True
    ImageArrayProxy = ArrayProxy

    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None, dtype=None):
        super().__init__(dataobj, affine, header, extra, file_map)
        self._header.set_data_offset(0)
        self._header.set_slope_inter(None, None)
        if dtype is not None:
            self.set_data_dtype(dtype)
    __init__.__doc__ = SpatialImage.__init__.__doc__

    def get_data_dtype(self):
        return self._header.get_data_dtype()

    def set_data_dtype(self, dtype):
        self._header.set_data_dtype(dtype)

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """Class method to create image from mapping in ``file_map``

        Parameters
        ----------
        file_map : dict
            Mapping with (kay, value) pairs of (``file_type``, FileHolder
            instance giving file-likes for each file needed for this image
            type.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_map`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.

        Returns
        -------
        img : AnalyzeImage instance
        """
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        hdr_fh, img_fh = klass._get_fileholders(file_map)
        with hdr_fh.get_prepare_fileobj(mode='rb') as hdrf:
            header = klass.header_class.from_fileobj(hdrf)
        hdr_copy = header.copy()
        imgf = img_fh.fileobj
        if imgf is None:
            imgf = img_fh.filename
        data = klass.ImageArrayProxy(imgf, hdr_copy, mmap=mmap, keep_file_open=keep_file_open)
        img = klass(data, None, header, file_map=file_map)
        img._affine = header.get_best_affine()
        img._load_cache = {'header': hdr_copy, 'affine': img._affine.copy(), 'file_map': copy_file_map(file_map)}
        return img

    @staticmethod
    def _get_fileholders(file_map):
        """Return fileholder for header and image

        Allows single-file image types to return one fileholder for both types.
        For Analyze there are two fileholders, one for the header, one for the
        image.
        """
        return (file_map['header'], file_map['image'])

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        dtype : dtype-like, optional
           The on-disk data type to coerce the data array.
        """
        if file_map is None:
            file_map = self.file_map
        data = np.asanyarray(self.dataobj)
        self.update_header()
        hdr = self._header
        offset = hdr.get_data_offset()
        data_dtype = hdr.get_data_dtype()
        if dtype is not None:
            hdr.set_data_dtype(dtype)
        out_dtype = hdr.get_data_dtype()
        slope = hdr['scl_slope'].item() if hdr.has_data_slope else np.nan
        inter = hdr['scl_inter'].item() if hdr.has_data_intercept else np.nan
        scale_me = np.all(np.isnan((slope, inter)))
        try:
            if scale_me:
                arr_writer = make_array_writer(data, out_dtype, hdr.has_data_slope, hdr.has_data_intercept)
            else:
                arr_writer = ArrayWriter(data, out_dtype, check_scaling=False)
        except WriterError:
            hdr.set_data_offset(offset)
            hdr.set_data_dtype(data_dtype)
            if hdr.has_data_slope:
                hdr['scl_slope'] = slope
            if hdr.has_data_intercept:
                hdr['scl_inter'] = inter
            raise
        hdr_fh, img_fh = self._get_fileholders(file_map)
        hdr_img_same = hdr_fh.same_file_as(img_fh)
        hdrf = hdr_fh.get_prepare_fileobj(mode='wb')
        if hdr_img_same:
            imgf = hdrf
        else:
            imgf = img_fh.get_prepare_fileobj(mode='wb')
        if scale_me:
            hdr.set_slope_inter(*get_slope_inter(arr_writer))
        hdr.write_to(hdrf)
        seek_tell(imgf, hdr.get_data_offset(), write0=True)
        arr_writer.to_fileobj(imgf)
        hdrf.close_if_mine()
        if not hdr_img_same:
            imgf.close_if_mine()
        self._header = hdr
        self.file_map = file_map
        hdr.set_data_offset(offset)
        hdr.set_data_dtype(data_dtype)
        if hdr.has_data_slope:
            hdr['scl_slope'] = slope
        if hdr.has_data_intercept:
            hdr['scl_inter'] = inter