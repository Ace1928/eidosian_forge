from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
def _data_tag_element(dataarray, encoding, dtype, ordering):
    """Creates data tag with given `encoding`, returns as XML element"""
    import zlib
    order = array_index_order_codes.npcode[ordering]
    enclabel = gifti_encoding_codes.label[encoding]
    if enclabel == 'ASCII':
        da = _arr2txt(dataarray, KIND2FMT[dtype.kind])
    elif enclabel in ('B64BIN', 'B64GZ'):
        out = np.asanyarray(dataarray, dtype).tobytes(order)
        if enclabel == 'B64GZ':
            out = zlib.compress(out)
        da = base64.b64encode(out).decode()
    elif enclabel == 'External':
        raise NotImplementedError('In what format are the external files?')
    else:
        da = ''
    data = xml.Element('Data')
    data.text = da
    return data