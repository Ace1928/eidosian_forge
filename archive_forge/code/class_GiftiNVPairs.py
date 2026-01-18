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
class GiftiNVPairs:
    """Gifti name / value pairs

    Attributes
    ----------
    name : str
    value : str
    """

    @deprecate_with_version('GiftiNVPairs objects are deprecated. Use the GiftiMetaData object as a dict, instead.', '4.0', '6.0')
    def __init__(self, name='', value=''):
        self._name = name
        self._value = value
        self._container = None

    @classmethod
    def _private_init(cls, name, value, md):
        """Private init method to provide warning-free experience"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            self = cls(name, value)
        self._container = md
        return self

    def __eq__(self, other):
        if not isinstance(other, GiftiNVPairs):
            return NotImplemented
        return self.name == other.name and self.value == other.value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, key):
        if self._container:
            self._container[key] = self._container.pop(self._name)
        self._name = key

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if self._container:
            self._container[self._name] = val
        self._value = val