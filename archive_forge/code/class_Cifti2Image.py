import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
class Cifti2Image(DataobjImage, SerializableImage):
    """Class for single file CIFTI-2 format image"""
    header_class = Cifti2Header
    header: Cifti2Header
    valid_exts = Nifti2Image.valid_exts
    files_types = Nifti2Image.files_types
    makeable = False
    rw = True

    def __init__(self, dataobj=None, header=None, nifti_header=None, extra=None, file_map=None, dtype=None):
        """Initialize image

        The image is a combination of (dataobj, header), with optional metadata
        in `nifti_header` (a NIfTI2 header).  There may be more metadata in the
        mapping `extra`. Filename / file-like objects can also go in the
        `file_map` mapping.

        Parameters
        ----------
        dataobj : object
            Object containing image data.  It should be some object that
            returns an array from ``np.asanyarray``.  It should have a
            ``shape`` attribute or property.
        header : Cifti2Header instance or sequence of :class:`cifti2_axes.Axis`
            Header with data for / from XML part of CIFTI-2 format.
            Alternatively a sequence of cifti2_axes.Axis objects can be provided
            describing each dimension of the array.
        nifti_header : None or mapping or NIfTI2 header instance, optional
            Metadata for NIfTI2 component of this format.
        extra : None or mapping
            Extra metadata not captured by `header` or `nifti_header`.
        file_map : mapping, optional
            Mapping giving file information for this image format.
        """
        if not isinstance(header, Cifti2Header) and header:
            header = Cifti2Header.from_axes(header)
        super().__init__(dataobj, header=header, extra=extra, file_map=file_map)
        self._nifti_header = LimitedNifti2Header.from_header(nifti_header)
        if dtype is not None:
            self.set_data_dtype(dtype)
        elif nifti_header is None and hasattr(dataobj, 'dtype'):
            self.set_data_dtype(dataobj.dtype)
        self.update_headers()
        if self._dataobj.shape != self.header.matrix.get_data_shape():
            warn(f'Dataobj shape {self._dataobj.shape} does not match shape expected from CIFTI-2 header {self.header.matrix.get_data_shape()}')

    @property
    def nifti_header(self):
        return self._nifti_header

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """Load a CIFTI-2 image from a file_map

        Parameters
        ----------
        file_map : file_map

        Returns
        -------
        img : Cifti2Image
            Returns a Cifti2Image
        """
        from .parse_cifti2 import Cifti2Extension, _Cifti2AsNiftiImage
        nifti_img = _Cifti2AsNiftiImage.from_file_map(file_map, mmap=mmap, keep_file_open=keep_file_open)
        for item in nifti_img.header.extensions:
            if isinstance(item, Cifti2Extension):
                cifti_header = item.get_content()
                break
        else:
            raise ValueError('NIfTI2 header does not contain a CIFTI-2 extension')
        dataobj = nifti_img.dataobj
        return Cifti2Image(reshape_dataobj(dataobj, dataobj.shape[4:]), header=cifti_header, nifti_header=nifti_img.header, file_map=file_map)

    @classmethod
    def from_image(klass, img):
        """Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : instance
            In fact, an object with the API of :class:`DataobjImage`.

        Returns
        -------
        cimg : instance
            Image, of our own class
        """
        if isinstance(img, klass):
            return img
        raise NotImplementedError

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead.

        Returns
        -------
        None
        """
        from .parse_cifti2 import Cifti2Extension
        self.update_headers()
        header = self._nifti_header
        extension = Cifti2Extension(content=self.header.to_xml())
        header.extensions = Nifti1Extensions((ext for ext in header.extensions if not isinstance(ext, Cifti2Extension)))
        header.extensions.append(extension)
        if self._dataobj.shape != self.header.matrix.get_data_shape():
            raise ValueError(f'Dataobj shape {self._dataobj.shape} does not match shape expected from CIFTI-2 header {self.header.matrix.get_data_shape()}')
        if header.get_intent()[0] == 'none':
            header.set_intent('NIFTI_INTENT_CONNECTIVITY_UNKNOWN')
        data = reshape_dataobj(self.dataobj, (1, 1, 1, 1) + self.dataobj.shape)
        if header['qform_code'] == 0:
            header['pixdim'][:4] = 1
        img = Nifti2Image(data, None, header, dtype=dtype)
        img.to_file_map(file_map or self.file_map)

    def update_headers(self):
        """Harmonize NIfTI headers with image data

        Ensures that the NIfTI-2 header records the data shape in the last three
        ``dim`` fields. Per the spec:

            Because the first four dimensions in NIfTI are reserved for space and time, the CIFTI
            dimensions are stored in the NIfTI header in dim[5] and up, where dim[5] is the length
            of the first CIFTI dimension (number of values in a row), dim[6] is the length of the
            second CIFTI dimension, and dim[7] is the length of the third CIFTI dimension, if
            applicable. The fields dim[1] through dim[4] will be 1; dim[0] will be 6 or 7,
            depending on whether a third matrix dimension exists.

        >>> import numpy as np
        >>> data = np.zeros((2,3,4))
        >>> img = Cifti2Image(data)  # doctest: +IGNORE_WARNINGS
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_headers()
        >>> img.nifti_header.get_data_shape() == (1, 1, 1, 1, 2, 3, 4)
        True
        >>> img.shape == (2, 3, 4)
        True
        """
        self._nifti_header.set_data_shape((1, 1, 1, 1) + self._dataobj.shape)

    def get_data_dtype(self):
        return self._nifti_header.get_data_dtype()

    def set_data_dtype(self, dtype):
        self._nifti_header.set_data_dtype(dtype)