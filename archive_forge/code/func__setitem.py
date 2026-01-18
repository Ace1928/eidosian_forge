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
def _setitem(self, tag, value, legacy_api):
    basetypes = (Number, bytes, str)
    info = TiffTags.lookup(tag, self.group)
    values = [value] if isinstance(value, basetypes) else value
    if tag not in self.tagtype:
        if info.type:
            self.tagtype[tag] = info.type
        else:
            self.tagtype[tag] = TiffTags.UNDEFINED
            if all((isinstance(v, IFDRational) for v in values)):
                self.tagtype[tag] = TiffTags.RATIONAL if all((v >= 0 for v in values)) else TiffTags.SIGNED_RATIONAL
            elif all((isinstance(v, int) for v in values)):
                if all((0 <= v < 2 ** 16 for v in values)):
                    self.tagtype[tag] = TiffTags.SHORT
                elif all((-2 ** 15 < v < 2 ** 15 for v in values)):
                    self.tagtype[tag] = TiffTags.SIGNED_SHORT
                else:
                    self.tagtype[tag] = TiffTags.LONG if all((v >= 0 for v in values)) else TiffTags.SIGNED_LONG
            elif all((isinstance(v, float) for v in values)):
                self.tagtype[tag] = TiffTags.DOUBLE
            elif all((isinstance(v, str) for v in values)):
                self.tagtype[tag] = TiffTags.ASCII
            elif all((isinstance(v, bytes) for v in values)):
                self.tagtype[tag] = TiffTags.BYTE
    if self.tagtype[tag] == TiffTags.UNDEFINED:
        values = [v.encode('ascii', 'replace') if isinstance(v, str) else v for v in values]
    elif self.tagtype[tag] == TiffTags.RATIONAL:
        values = [float(v) if isinstance(v, int) else v for v in values]
    is_ifd = self.tagtype[tag] == TiffTags.LONG and isinstance(values, dict)
    if not is_ifd:
        values = tuple((info.cvt_enum(value) for value in values))
    dest = self._tags_v1 if legacy_api else self._tags_v2
    if not is_ifd and (info.length == 1 or self.tagtype[tag] == TiffTags.BYTE or (info.length is None and len(values) == 1 and (not legacy_api))):
        if legacy_api and self.tagtype[tag] in [TiffTags.RATIONAL, TiffTags.SIGNED_RATIONAL]:
            values = (values,)
        try:
            dest[tag], = values
        except ValueError:
            warnings.warn(f'Metadata Warning, tag {tag} had too many entries: {len(values)}, expected 1')
            dest[tag] = values[0]
    else:
        dest[tag] = values