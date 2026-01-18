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
def _writeGlyphToBytes(glyphName, glyphObject=None, drawPointsFunc=None, writer=None, formatVersion=None, validate=True):
    """Return .glif data for a glyph as a UTF-8 encoded bytes string."""
    try:
        formatVersion = GLIFFormatVersion(formatVersion)
    except ValueError:
        from fontTools.ufoLib.errors import UnsupportedGLIFFormat
        raise UnsupportedGLIFFormat('Unsupported GLIF format version: {formatVersion!r}')
    if validate and (not isinstance(glyphName, str)):
        raise GlifLibError('The glyph name is not properly formatted.')
    if validate and len(glyphName) == 0:
        raise GlifLibError('The glyph name is empty.')
    glyphAttrs = OrderedDict([('name', glyphName), ('format', repr(formatVersion.major))])
    if formatVersion.minor != 0:
        glyphAttrs['formatMinor'] = repr(formatVersion.minor)
    root = etree.Element('glyph', glyphAttrs)
    identifiers = set()
    _writeAdvance(glyphObject, root, validate)
    if getattr(glyphObject, 'unicodes', None):
        _writeUnicodes(glyphObject, root, validate)
    if getattr(glyphObject, 'note', None):
        _writeNote(glyphObject, root, validate)
    if formatVersion.major >= 2 and getattr(glyphObject, 'image', None):
        _writeImage(glyphObject, root, validate)
    if formatVersion.major >= 2 and getattr(glyphObject, 'guidelines', None):
        _writeGuidelines(glyphObject, root, identifiers, validate)
    anchors = getattr(glyphObject, 'anchors', None)
    if formatVersion.major >= 2 and anchors:
        _writeAnchors(glyphObject, root, identifiers, validate)
    if drawPointsFunc is not None:
        outline = etree.SubElement(root, 'outline')
        pen = GLIFPointPen(outline, identifiers=identifiers, validate=validate)
        drawPointsFunc(pen)
        if formatVersion.major == 1 and anchors:
            _writeAnchorsFormat1(pen, anchors, validate)
        if not len(outline):
            outline.text = '\n  '
    if getattr(glyphObject, 'lib', None):
        _writeLib(glyphObject, root, validate)
    data = etree.tostring(root, encoding='UTF-8', xml_declaration=True, pretty_print=True)
    return data