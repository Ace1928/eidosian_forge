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
class Nifti1Extension:
    """Baseclass for NIfTI1 header extensions.

    This class is sufficient to handle very simple text-based extensions, such
    as `comment`. More sophisticated extensions should/will be supported by
    dedicated subclasses.
    """

    def __init__(self, code, content):
        """
        Parameters
        ----------
        code : int or str
          Canonical extension code as defined in the NIfTI standard, given
          either as integer or corresponding label
          (see :data:`~nibabel.nifti1.extension_codes`)
        content : str
          Extension content as read from the NIfTI file header. This content is
          converted into a runtime representation.
        """
        try:
            self._code = extension_codes.code[code]
        except KeyError:
            self._code = code
        self._content = self._unmangle(content)

    def _unmangle(self, value):
        """Convert the extension content into its runtime representation.

        The default implementation does nothing at all.

        Parameters
        ----------
        value : str
          Extension content as read from file.

        Returns
        -------
        The same object that was passed as `value`.

        Notes
        -----
        Subclasses should reimplement this method to provide the desired
        unmangling procedure and may return any type of object.
        """
        return value

    def _mangle(self, value):
        """Convert the extension content into NIfTI file header representation.

        The default implementation does nothing at all.

        Parameters
        ----------
        value : str
          Extension content in runtime form.

        Returns
        -------
        str

        Notes
        -----
        Subclasses should reimplement this method to provide the desired
        mangling procedure.
        """
        return value

    def get_code(self):
        """Return the canonical extension type code."""
        return self._code

    def get_content(self):
        """Return the extension content in its runtime representation."""
        return self._content

    def get_sizeondisk(self):
        """Return the size of the extension in the NIfTI file."""
        size = len(self._mangle(self._content))
        size += 8
        if size % 16 != 0:
            size += 16 - size % 16
        return size

    def __repr__(self):
        try:
            code = extension_codes.label[self._code]
        except KeyError:
            code = self._code
        s = f"Nifti1Extension('{code}', '{self._content}')"
        return s

    def __eq__(self, other):
        return (self._code, self._content) == (other._code, other._content)

    def __ne__(self, other):
        return not self == other

    def write_to(self, fileobj, byteswap):
        """Write header extensions to fileobj

        Write starts at fileobj current file position.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` method
        byteswap : boolean
          Flag if byteswapping the data is required.

        Returns
        -------
        None
        """
        extstart = fileobj.tell()
        rawsize = self.get_sizeondisk()
        extinfo = np.array((rawsize, self._code), dtype=np.int32)
        if byteswap:
            extinfo = extinfo.byteswap()
        fileobj.write(extinfo.tobytes())
        fileobj.write(self._mangle(self._content))
        fileobj.write(b'\x00' * (extstart + rawsize - fileobj.tell()))