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
def _buildOutlineContourFormat2(pen, contour, identifiers, validate):
    if validate:
        for attr in contour.attrib.keys():
            if attr not in contourAttributesFormat2:
                raise GlifLibError('Unknown attribute in contour element: %s' % attr)
    identifier = contour.get('identifier')
    if identifier is not None:
        if validate:
            if identifier in identifiers:
                raise GlifLibError('The identifier %s is used more than once.' % identifier)
            if not identifierValidator(identifier):
                raise GlifLibError('The contour identifier %s is not valid.' % identifier)
        identifiers.add(identifier)
    try:
        pen.beginPath(identifier=identifier)
    except TypeError:
        pen.beginPath()
        warn("The beginPath method needs an identifier kwarg. The contour's identifier value has been discarded.", DeprecationWarning)
    if len(contour):
        massaged = _validateAndMassagePointStructures(contour, pointAttributesFormat2, validate=validate)
        _buildOutlinePointsFormat2(pen, massaged, identifiers, validate)
    pen.endPath()