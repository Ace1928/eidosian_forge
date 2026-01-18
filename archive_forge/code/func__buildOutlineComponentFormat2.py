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
def _buildOutlineComponentFormat2(pen, component, identifiers, validate):
    if validate:
        if len(component):
            raise GlifLibError('Unknown child elements of component element.')
        for attr in component.attrib.keys():
            if attr not in componentAttributesFormat2:
                raise GlifLibError('Unknown attribute in component element: %s' % attr)
    baseGlyphName = component.get('base')
    if validate and baseGlyphName is None:
        raise GlifLibError('The base attribute is not defined in the component.')
    transformation = []
    for attr, default in _transformationInfo:
        value = component.get(attr)
        if value is None:
            value = default
        else:
            value = _number(value)
        transformation.append(value)
    identifier = component.get('identifier')
    if identifier is not None:
        if validate:
            if identifier in identifiers:
                raise GlifLibError('The identifier %s is used more than once.' % identifier)
            if validate and (not identifierValidator(identifier)):
                raise GlifLibError('The identifier %s is not valid.' % identifier)
        identifiers.add(identifier)
    try:
        pen.addComponent(baseGlyphName, tuple(transformation), identifier=identifier)
    except TypeError:
        pen.addComponent(baseGlyphName, tuple(transformation))
        warn("The addComponent method needs an identifier kwarg. The component's identifier value has been discarded.", DeprecationWarning)