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
def _writeImage(glyphObject, element, validate):
    image = getattr(glyphObject, 'image', None)
    if validate and (not imageValidator(image)):
        raise GlifLibError('image attribute must be a dict or dict-like object with the proper structure.')
    attrs = OrderedDict([('fileName', image['fileName'])])
    for attr, default in _transformationInfo:
        value = image.get(attr, default)
        if value != default:
            attrs[attr] = repr(value)
    color = image.get('color')
    if color is not None:
        attrs['color'] = color
    etree.SubElement(element, 'image', attrs)