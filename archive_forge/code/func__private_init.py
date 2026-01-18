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
@classmethod
def _private_init(cls, name, value, md):
    """Private init method to provide warning-free experience"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        self = cls(name, value)
    self._container = md
    return self