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
def _number(s):
    """
    Given a numeric string, return an integer or a float, whichever
    the string indicates. _number("1") will return the integer 1,
    _number("1.0") will return the float 1.0.

    >>> _number("1")
    1
    >>> _number("1.0")
    1.0
    >>> _number("a")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    GlifLibError: Could not convert a to an int or float.
    """
    try:
        n = int(s)
        return n
    except ValueError:
        pass
    try:
        n = float(s)
        return n
    except ValueError:
        raise GlifLibError('Could not convert %s to an int or float.' % s)