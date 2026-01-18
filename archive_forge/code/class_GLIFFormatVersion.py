from __future__ import annotations
import logging
import enum
from warnings import warn
from collections import OrderedDict
import fs
import fs.base
import fs.errors
import fs.osfs
import fs.path
from fontTools.misc.textTools import tobytes
from fontTools.misc import plistlib
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from fontTools.ufoLib.errors import GlifLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.validators import (
from fontTools.misc import etree
from fontTools.ufoLib import _UFOBaseIO, UFOFormatVersion
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
class GLIFFormatVersion(tuple, _VersionTupleEnumMixin, enum.Enum):
    FORMAT_1_0 = (1, 0)
    FORMAT_2_0 = (2, 0)

    @classmethod
    def default(cls, ufoFormatVersion=None):
        if ufoFormatVersion is not None:
            return max(cls.supported_versions(ufoFormatVersion))
        return super().default()

    @classmethod
    def supported_versions(cls, ufoFormatVersion=None):
        if ufoFormatVersion is None:
            return super().supported_versions()
        versions = {cls.FORMAT_1_0}
        if ufoFormatVersion >= UFOFormatVersion.FORMAT_3_0:
            versions.add(cls.FORMAT_2_0)
        return frozenset(versions)