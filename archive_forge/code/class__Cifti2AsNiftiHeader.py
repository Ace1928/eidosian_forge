from io import BytesIO
import numpy as np
from packaging.version import Version, parse
from .. import xmlutils as xml
from ..batteryrunners import Report
from ..nifti1 import Nifti1Extension, extension_codes, intent_codes
from ..nifti2 import Nifti2Header, Nifti2Image
from ..spatialimages import HeaderDataError
from .cifti2 import (
class _Cifti2AsNiftiHeader(Nifti2Header):
    """Class for Cifti2 header extension"""

    @classmethod
    def _valid_intent_code(klass, intent_code):
        """Return True if `intent_code` matches our class `klass`"""
        return intent_code >= 3000 and intent_code < 3100

    @classmethod
    def may_contain_header(klass, binaryblock):
        if not super().may_contain_header(binaryblock):
            return False
        hdr = klass(binaryblock=binaryblock[:klass.sizeof_hdr])
        return klass._valid_intent_code(hdr.get_intent('code')[0])

    @staticmethod
    def _chk_qfac(hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['pixdim'][0] in (-1, 0, 1):
            return (hdr, rep)
        rep.problem_level = 20
        rep.problem_msg = 'pixdim[0] (qfac) should be 1 (default) or 0 or -1'
        if fix:
            hdr['pixdim'][0] = 1
            rep.fix_msg = 'setting qfac to 1'
        return (hdr, rep)

    @staticmethod
    def _chk_pixdims(hdr, fix=False):
        rep = Report(HeaderDataError)
        pixdims = hdr['pixdim']
        spat_dims = pixdims[1:4]
        if not np.any(spat_dims < 0):
            return (hdr, rep)
        rep.problem_level = 35
        rep.problem_msg = 'pixdim[1,2,3] should be zero or positive'
        if fix:
            hdr['pixdim'][1:4] = np.abs(spat_dims)
            rep.fix_msg = 'setting to abs of pixdim values'
        return (hdr, rep)