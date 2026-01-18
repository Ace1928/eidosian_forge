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
class _FetchUnicodesParser(_BaseParser):

    def __init__(self):
        self.unicodes = []
        super().__init__()

    def startElementHandler(self, name, attrs):
        if name == 'unicode' and self._elementStack and (self._elementStack[-1] == 'glyph'):
            value = attrs.get('hex')
            if value is not None:
                try:
                    value = int(value, 16)
                    if value not in self.unicodes:
                        self.unicodes.append(value)
                except ValueError:
                    pass
        super().startElementHandler(name, attrs)