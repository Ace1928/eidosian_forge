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
def _writeAdvance(glyphObject, element, validate):
    width = getattr(glyphObject, 'width', None)
    if width is not None:
        if validate and (not isinstance(width, numberTypes)):
            raise GlifLibError('width attribute must be int or float')
        if width == 0:
            width = None
    height = getattr(glyphObject, 'height', None)
    if height is not None:
        if validate and (not isinstance(height, numberTypes)):
            raise GlifLibError('height attribute must be int or float')
        if height == 0:
            height = None
    if width is not None and height is not None:
        etree.SubElement(element, 'advance', OrderedDict([('height', repr(height)), ('width', repr(width))]))
    elif width is not None:
        etree.SubElement(element, 'advance', dict(width=repr(width)))
    elif height is not None:
        etree.SubElement(element, 'advance', dict(height=repr(height)))