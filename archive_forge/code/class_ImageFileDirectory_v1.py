from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
class ImageFileDirectory_v1(ImageFileDirectory_v2):
    """This class represents the **legacy** interface to a TIFF tag directory.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v1()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        ('Some Data',)

    Also contains a dictionary of tag types as read from the tiff image file,
    :attr:`~PIL.TiffImagePlugin.ImageFileDirectory_v1.tagtype`.

    Values are returned as a tuple.

    ..  deprecated:: 3.0.0
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._legacy_api = True
    tags = property(lambda self: self._tags_v1)
    tagdata = property(lambda self: self._tagdata)
    tagtype: dict
    'Dictionary of tag types'

    @classmethod
    def from_v2(cls, original):
        """Returns an
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`
        instance with the same data as is contained in the original
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`
        instance.

        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`

        """
        ifd = cls(prefix=original.prefix)
        ifd._tagdata = original._tagdata
        ifd.tagtype = original.tagtype
        ifd.next = original.next
        return ifd

    def to_v2(self):
        """Returns an
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`
        instance with the same data as is contained in the original
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`
        instance.

        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`

        """
        ifd = ImageFileDirectory_v2(prefix=self.prefix)
        ifd._tagdata = dict(self._tagdata)
        ifd.tagtype = dict(self.tagtype)
        ifd._tags_v2 = dict(self._tags_v2)
        return ifd

    def __contains__(self, tag):
        return tag in self._tags_v1 or tag in self._tagdata

    def __len__(self):
        return len(set(self._tagdata) | set(self._tags_v1))

    def __iter__(self):
        return iter(set(self._tagdata) | set(self._tags_v1))

    def __setitem__(self, tag, value):
        for legacy_api in (False, True):
            self._setitem(tag, value, legacy_api)

    def __getitem__(self, tag):
        if tag not in self._tags_v1:
            data = self._tagdata[tag]
            typ = self.tagtype[tag]
            size, handler = self._load_dispatch[typ]
            for legacy in (False, True):
                self._setitem(tag, handler(self, data, legacy), legacy)
        val = self._tags_v1[tag]
        if not isinstance(val, (tuple, bytes)):
            val = (val,)
        return val