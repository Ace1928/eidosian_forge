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
def _readGlyphFromTree(tree, glyphObject=None, pointPen=None, formatVersions=GLIFFormatVersion.supported_versions(), validate=True):
    formatVersionMajor = tree.get('format')
    if validate and formatVersionMajor is None:
        raise GlifLibError('Unspecified format version in GLIF.')
    formatVersionMinor = tree.get('formatMinor', 0)
    try:
        formatVersion = GLIFFormatVersion((int(formatVersionMajor), int(formatVersionMinor)))
    except ValueError as e:
        msg = 'Unsupported GLIF format: %s.%s' % (formatVersionMajor, formatVersionMinor)
        if validate:
            from fontTools.ufoLib.errors import UnsupportedGLIFFormat
            raise UnsupportedGLIFFormat(msg) from e
        formatVersion = GLIFFormatVersion.default()
        logger.warning('%s. Assuming the latest supported version (%s). Some data may be skipped or parsed incorrectly.', msg, formatVersion)
    if validate and formatVersion not in formatVersions:
        raise GlifLibError(f'Forbidden GLIF format version: {formatVersion!s}')
    try:
        readGlyphFromTree = _READ_GLYPH_FROM_TREE_FUNCS[formatVersion]
    except KeyError:
        raise NotImplementedError(formatVersion)
    readGlyphFromTree(tree=tree, glyphObject=glyphObject, pointPen=pointPen, validate=validate, formatMinor=formatVersion.minor)