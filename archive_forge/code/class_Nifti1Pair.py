from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
class Nifti1Pair(analyze.AnalyzeImage):
    """Class for NIfTI1 format image, header pair"""
    header_class: type[Nifti1Header] = Nifti1PairHeader
    header: Nifti1Header
    _meta_sniff_len = header_class.sizeof_hdr
    rw = True
    _dtype_alias = None

    def __init__(self, dataobj, affine, header=None, extra=None, file_map=None, dtype=None):
        danger_dts = (np.dtype('int64'), np.dtype('uint64'))
        if header is None and dtype is None and (get_obj_dtype(dataobj) in danger_dts):
            alert_future_error(f'Image data has type {dataobj.dtype}, which may cause incompatibilities with other tools.', '5.0', warning_rec=f'This warning can be silenced by passing the dtype argument to {self.__class__.__name__}().', error_rec=f'To use this type, pass an explicit header or dtype argument to {self.__class__.__name__}().', error_class=ValueError)
        super().__init__(dataobj, affine, header, extra, file_map, dtype)
        if header is None and affine is not None:
            self._affine2header()
    __init__.__doc__ = f'{analyze.AnalyzeImage.__init__.__doc__}\n        Notes\n        -----\n\n        If both a `header` and an `affine` are specified, and the `affine` does\n        not match the affine that is in the `header`, the `affine` will be used,\n        but the ``sform_code`` and ``qform_code`` fields in the header will be\n        re-initialised to their default values. This is performed on the basis\n        that, if you are changing the affine, you are likely to be changing the\n        space to which the affine is pointing.  The :meth:`set_sform` and\n        :meth:`set_qform` methods can be used to update the codes after an image\n        has been created - see those methods, and the :ref:`manual\n        <default-sform-qform-codes>` for more details.  '

    def update_header(self):
        """Harmonize header with image data and affine

        See AnalyzeImage.update_header for more examples

        Examples
        --------
        >>> data = np.zeros((2,3,4))
        >>> affine = np.diag([1.0,2.0,3.0,1.0])
        >>> img = Nifti1Image(data, affine)
        >>> hdr = img.header
        >>> np.all(hdr.get_qform() == affine)
        True
        >>> np.all(hdr.get_sform() == affine)
        True
        """
        super().update_header()
        hdr = self._header
        hdr['magic'] = hdr.pair_magic

    def _affine2header(self):
        """Unconditionally set affine into the header"""
        hdr = self._header
        hdr.set_sform(self._affine, code='aligned')
        hdr.set_qform(self._affine, code='unknown')

    def get_qform(self, coded=False):
        """Return 4x4 affine matrix from qform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and qform code.  If False, just
            return affine.  {affine or None} means, return None if qform code
            == 0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine reconstructed from qform
            quaternion.  If `coded` is True, return None if qform code is 0,
            else return the affine.
        code : int
            Qform code. Only returned if `coded` is True.

        See also
        --------
        set_qform
        get_sform
        """
        return self._header.get_qform(coded)

    def set_qform(self, affine, code=None, strip_shears=True, **kwargs):
        """Set qform header values from 4x4 affine

        Parameters
        ----------
        affine : None or 4x4 array
            affine transform to write into sform. If None, only set code.
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:

            * If affine is None, `code`-> 0
            * If affine not None and existing qform code in header == 0,
              `code`-> 2 (aligned)
            * If affine not None and existing qform code in header != 0,
              `code`-> existing qform code in header

        strip_shears : bool, optional
            Whether to strip shears in `affine`.  If True, shears will be
            silently stripped. If False, the presence of shears will raise a
            ``HeaderDataError``
        update_affine : bool, optional
            Whether to update the image affine from the header best affine
            after setting the qform. Must be keyword argument (because of
            different position in `set_qform`). Default is True

        See also
        --------
        get_qform
        set_sform

        Examples
        --------
        >>> data = np.arange(24, dtype='f4').reshape((2,3,4))
        >>> aff = np.diag([2, 3, 4, 1])
        >>> img = Nifti1Pair(data, aff)
        >>> img.get_qform()
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> img.get_qform(coded=True)
        (None, 0)
        >>> aff2 = np.diag([3, 4, 5, 1])
        >>> img.set_qform(aff2, 'talairach')
        >>> qaff, code = img.get_qform(coded=True)
        >>> np.all(qaff == aff2)
        True
        >>> int(code)
        3
        """
        update_affine = kwargs.pop('update_affine', True)
        if kwargs:
            raise TypeError(f'Unexpected keyword argument(s) {kwargs}')
        self._header.set_qform(affine, code, strip_shears)
        if update_affine:
            if self._affine is None:
                self._affine = self._header.get_best_affine()
            else:
                self._affine[:] = self._header.get_best_affine()

    def get_sform(self, coded=False):
        """Return 4x4 affine matrix from sform parameters in header

        Parameters
        ----------
        coded : bool, optional
            If True, return {affine or None}, and sform code.  If False, just
            return affine.  {affine or None} means, return None if sform code
            == 0, and affine otherwise.

        Returns
        -------
        affine : None or (4,4) ndarray
            If `coded` is False, always return affine from sform fields. If
            `coded` is True, return None if sform code is 0, else return the
            affine.
        code : int
            Sform code. Only returned if `coded` is True.

        See also
        --------
        set_sform
        get_qform
        """
        return self._header.get_sform(coded)

    def set_sform(self, affine, code=None, **kwargs):
        """Set sform transform from 4x4 affine

        Parameters
        ----------
        affine : None or 4x4 array
            affine transform to write into sform.  If None, only set `code`
        code : None, string or integer
            String or integer giving meaning of transform in *affine*.
            The default is None.  If code is None, then:

            * If affine is None, `code`-> 0
            * If affine not None and existing sform code in header == 0,
              `code`-> 2 (aligned)
            * If affine not None and existing sform code in header != 0,
              `code`-> existing sform code in header

        update_affine : bool, optional
            Whether to update the image affine from the header best affine
            after setting the qform.  Must be keyword argument (because of
            different position in `set_qform`). Default is True

        See also
        --------
        get_sform
        set_qform

        Examples
        --------
        >>> data = np.arange(24, dtype='f4').reshape((2,3,4))
        >>> aff = np.diag([2, 3, 4, 1])
        >>> img = Nifti1Pair(data, aff)
        >>> img.get_sform()
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> saff, code = img.get_sform(coded=True)
        >>> saff
        array([[2., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 4., 0.],
               [0., 0., 0., 1.]])
        >>> int(code)
        2
        >>> aff2 = np.diag([3, 4, 5, 1])
        >>> img.set_sform(aff2, 'talairach')
        >>> saff, code = img.get_sform(coded=True)
        >>> np.all(saff == aff2)
        True
        >>> int(code)
        3
        """
        update_affine = kwargs.pop('update_affine', True)
        if kwargs:
            raise TypeError(f'Unexpected keyword argument(s) {kwargs}')
        self._header.set_sform(affine, code)
        if update_affine:
            if self._affine is None:
                self._affine = self._header.get_best_affine()
            else:
                self._affine[:] = self._header.get_best_affine()

    def set_data_dtype(self, datatype):
        """Set numpy dtype for data from code, dtype, type or alias

        Using :py:class:`int` or ``"int"`` is disallowed, as these types
        will be interpreted as ``np.int64``, which is almost never desired.
        ``np.int64`` is permitted for those intent on making poor choices.

        The following aliases are defined to allow for flexible specification:

          * ``'mask'`` - Alias for ``uint8``
          * ``'compat'`` - The nearest Analyze-compatible datatype
            (``uint8``, ``int16``, ``int32``, ``float32``)
          * ``'smallest'`` - The smallest Analyze-compatible integer
            (``uint8``, ``int16``, ``int32``)

        Dynamic aliases are resolved when ``get_data_dtype()`` is called
        with a ``finalize=True`` flag. Until then, these aliases are not
        written to the header and will not persist to new images.

        Examples
        --------
        >>> ints = np.arange(24, dtype='i4').reshape((2,3,4))

        >>> img = Nifti1Image(ints, np.eye(4))
        >>> img.set_data_dtype(np.uint8)
        >>> img.get_data_dtype()
        dtype('uint8')
        >>> img.set_data_dtype('mask')
        >>> img.get_data_dtype()
        dtype('uint8')
        >>> img.set_data_dtype('compat')
        >>> img.get_data_dtype()
        'compat'
        >>> img.get_data_dtype(finalize=True)
        dtype('<i4')
        >>> img.get_data_dtype()
        dtype('<i4')
        >>> img.set_data_dtype('smallest')
        >>> img.get_data_dtype()
        'smallest'
        >>> img.get_data_dtype(finalize=True)
        dtype('uint8')
        >>> img.get_data_dtype()
        dtype('uint8')

        Note that floating point values will not be coerced to ``int``

        >>> floats = np.arange(24, dtype='f4').reshape((2,3,4))
        >>> img = Nifti1Image(floats, np.eye(4))
        >>> img.set_data_dtype('smallest')
        >>> img.get_data_dtype(finalize=True)
        Traceback (most recent call last):
           ...
        ValueError: Cannot automatically cast array (of type float32) to an integer
        type with fewer than 64 bits. Please set_data_dtype() to an explicit data type.

        >>> arr = np.arange(1000, 1024, dtype='i4').reshape((2,3,4))
        >>> img = Nifti1Image(arr, np.eye(4))
        >>> img.set_data_dtype('smallest')
        >>> img.set_data_dtype('implausible')
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "implausible" not recognized
        >>> img.set_data_dtype('none')
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "none" known but not supported
        >>> img.set_data_dtype(np.void)
        Traceback (most recent call last):
           ...
        nibabel.spatialimages.HeaderDataError: data dtype "<class 'numpy.void'>" known
        but not supported
        >>> img.set_data_dtype('int')
        Traceback (most recent call last):
           ...
        ValueError: Invalid data type 'int'. Specify a sized integer, e.g., 'uint8' or numpy.int16.
        >>> img.set_data_dtype(int)
        Traceback (most recent call last):
           ...
        ValueError: Invalid data type <class 'int'>. Specify a sized integer, e.g., 'uint8' or
        numpy.int16.
        >>> img.set_data_dtype('int64')
        >>> img.get_data_dtype() == np.dtype('int64')
        True
        """
        if isinstance(datatype, str):
            if datatype == 'mask':
                datatype = 'u1'
            elif datatype in ('compat', 'smallest'):
                self._dtype_alias = datatype
                return
        self._dtype_alias = None
        super().set_data_dtype(datatype)

    def get_data_dtype(self, finalize=False):
        """Get numpy dtype for data

        If ``set_data_dtype()`` has been called with an alias
        and ``finalize`` is ``False``, return the alias.
        If ``finalize`` is ``True``, determine the appropriate dtype
        from the image data object and set the final dtype in the
        header before returning it.
        """
        if self._dtype_alias is None:
            return super().get_data_dtype()
        if not finalize:
            return self._dtype_alias
        datatype = None
        if self._dtype_alias == 'compat':
            datatype = _get_analyze_compat_dtype(self._dataobj)
            descrip = 'an Analyze-compatible dtype'
        elif self._dtype_alias == 'smallest':
            datatype = _get_smallest_dtype(self._dataobj)
            descrip = 'an integer type with fewer than 64 bits'
        else:
            raise ValueError(f'Unknown dtype alias {self._dtype_alias}.')
        if datatype is None:
            dt = get_obj_dtype(self._dataobj)
            raise ValueError(f'Cannot automatically cast array (of type {dt}) to {descrip}. Please set_data_dtype() to an explicit data type.')
        self.set_data_dtype(datatype)
        return super().get_data_dtype()

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
        img_dtype = self.get_data_dtype()
        self.get_data_dtype(finalize=True)
        try:
            super().to_file_map(file_map, dtype)
        finally:
            self.set_data_dtype(img_dtype)

    def as_reoriented(self, ornt):
        """Apply an orientation change and return a new image

        If ornt is identity transform, return the original image, unchanged

        Parameters
        ----------
        ornt : (n,2) orientation array
           orientation transform. ``ornt[N,1]` is flip of axis N of the
           array implied by `shape`, where 1 means no flip and -1 means
           flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
           there's an array ``arr`` of shape `shape`, the flip would
           correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
           the transpose that needs to be done to the implied array, as in
           ``arr.transpose(ornt[:,0])``
        """
        img = super().as_reoriented(ornt)
        if img is self:
            return img
        new_dim = [None if orig_dim is None else int(ornt[orig_dim, 0]) for orig_dim in img.header.get_dim_info()]
        img.header.set_dim_info(*new_dim)
        return img