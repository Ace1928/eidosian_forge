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
def _readLib(glyphObject, lib, validate):
    assert len(lib) == 1
    child = lib[0]
    plist = plistlib.fromtree(child)
    if validate:
        valid, message = glyphLibValidator(plist)
        if not valid:
            raise GlifLibError(message)
    _relaxedSetattr(glyphObject, 'lib', plist)