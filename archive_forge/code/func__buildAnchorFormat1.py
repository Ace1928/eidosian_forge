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
def _buildAnchorFormat1(point, validate):
    if point.get('type') != 'move':
        return None
    name = point.get('name')
    if name is None:
        return None
    x = point.get('x')
    y = point.get('y')
    if validate and x is None:
        raise GlifLibError('Required x attribute is missing in point element.')
    if validate and y is None:
        raise GlifLibError('Required y attribute is missing in point element.')
    x = _number(x)
    y = _number(y)
    anchor = dict(x=x, y=y, name=name)
    return anchor