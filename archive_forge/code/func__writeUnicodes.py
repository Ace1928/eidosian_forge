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
def _writeUnicodes(glyphObject, element, validate):
    unicodes = getattr(glyphObject, 'unicodes', None)
    if validate and isinstance(unicodes, int):
        unicodes = [unicodes]
    seen = set()
    for code in unicodes:
        if validate and (not isinstance(code, int)):
            raise GlifLibError('unicode values must be int')
        if code in seen:
            continue
        seen.add(code)
        hexCode = '%04X' % code
        etree.SubElement(element, 'unicode', dict(hex=hexCode))