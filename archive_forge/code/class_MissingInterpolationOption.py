import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
class MissingInterpolationOption(InterpolationError):
    """A value specified for interpolation was missing."""

    def __init__(self, option):
        msg = 'missing option "%s" in interpolation.' % option
        InterpolationError.__init__(self, msg)