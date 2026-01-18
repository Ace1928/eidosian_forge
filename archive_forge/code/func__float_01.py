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
def _float_01(val):
    out = float(val)
    if out < 0 or out > 1:
        raise ValueError('Float must be between 0 and 1 inclusive')
    return out