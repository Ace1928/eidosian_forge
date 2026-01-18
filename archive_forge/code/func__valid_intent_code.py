from io import BytesIO
import numpy as np
from packaging.version import Version, parse
from .. import xmlutils as xml
from ..batteryrunners import Report
from ..nifti1 import Nifti1Extension, extension_codes, intent_codes
from ..nifti2 import Nifti2Header, Nifti2Image
from ..spatialimages import HeaderDataError
from .cifti2 import (
@classmethod
def _valid_intent_code(klass, intent_code):
    """Return True if `intent_code` matches our class `klass`"""
    return intent_code >= 3000 and intent_code < 3100