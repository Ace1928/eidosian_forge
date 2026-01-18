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
class Nifti1DicomExtension(Nifti1Extension):
    """NIfTI1 DICOM header extension

    This class is a thin wrapper around pydicom to read a binary DICOM
    byte string. If pydicom is available, content is exposed as a Dicom Dataset.
    Otherwise, this silently falls back to the standard NiftiExtension class
    and content is the raw bytestring loaded directly from the nifti file
    header.
    """

    def __init__(self, code, content, parent_hdr=None):
        """
        Parameters
        ----------
        code : int or str
          Canonical extension code as defined in the NIfTI standard, given
          either as integer or corresponding label
          (see :data:`~nibabel.nifti1.extension_codes`)
        content : bytes or pydicom Dataset or None
          Extension content - either a bytestring as read from the NIfTI file
          header or an existing pydicom Dataset. If a bystestring, the content
          is converted into a Dataset on initialization. If None, a new empty
          Dataset is created.
        parent_hdr : :class:`~nibabel.nifti1.Nifti1Header`, optional
          If a dicom extension belongs to an existing
          :class:`~nibabel.nifti1.Nifti1Header`, it may be provided here to
          ensure that the DICOM dataset is written with correctly corresponding
          endianness; otherwise it is assumed the dataset is little endian.

        Notes
        -----

        code should always be 2 for DICOM.
        """
        self._code = code
        if parent_hdr:
            self._is_little_endian = parent_hdr.endianness == '<'
        else:
            self._is_little_endian = True
        if isinstance(content, pdcm.dataset.Dataset):
            self._is_implicit_VR = False
            self._raw_content = self._mangle(content)
            self._content = content
        elif isinstance(content, bytes):
            self._raw_content = content
            self._is_implicit_VR = self._guess_implicit_VR()
            ds = self._unmangle(content, self._is_implicit_VR, self._is_little_endian)
            self._content = ds
        elif content is None:
            self._is_implicit_VR = False
            self._content = pdcm.dataset.Dataset()
        else:
            raise TypeError(f'content must be either a bytestring or a pydicom Dataset. Got {content.__class__}')

    def _guess_implicit_VR(self):
        """Try to guess DICOM syntax by checking for valid VRs.

        Without a DICOM Transfer Syntax, it's difficult to tell if Value
        Representations (VRs) are included in the DICOM encoding or not.
        This reads where the first VR would be and checks it against a list of
        valid VRs
        """
        potential_vr = self._raw_content[4:6].decode()
        if potential_vr in pdcm.values.converters.keys():
            implicit_VR = False
        else:
            implicit_VR = True
        return implicit_VR

    def _unmangle(self, value, is_implicit_VR=False, is_little_endian=True):
        bio = BytesIO(value)
        ds = pdcm.filereader.read_dataset(bio, is_implicit_VR, is_little_endian)
        return ds

    def _mangle(self, dataset):
        bio = BytesIO()
        dio = pdcm.filebase.DicomFileLike(bio)
        dio.is_implicit_VR = self._is_implicit_VR
        dio.is_little_endian = self._is_little_endian
        ds_len = pdcm.filewriter.write_dataset(dio, dataset)
        dio.seek(0)
        return dio.read(ds_len)