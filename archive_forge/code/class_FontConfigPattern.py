from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
class FontConfigPattern:

    def __init__(self, fontconfig, pattern=None):
        self._fontconfig = fontconfig
        self._pattern = pattern

    @property
    def is_valid(self):
        return self._fontconfig and self._pattern

    def _create(self):
        assert not self._pattern
        assert self._fontconfig
        self._pattern = self._fontconfig.FcPatternCreate()

    def _destroy(self):
        assert self._pattern
        assert self._fontconfig
        self._fontconfig.FcPatternDestroy(self._pattern)
        self._pattern = None

    @staticmethod
    def _bold_to_weight(bold):
        return FC_WEIGHT_BOLD if bold else FC_WEIGHT_REGULAR

    @staticmethod
    def _italic_to_slant(italic):
        return FC_SLANT_ITALIC if italic else FC_SLANT_ROMAN

    def _set_string(self, name, value):
        assert self._pattern
        assert name
        assert self._fontconfig
        if not value:
            return
        value = value.encode('utf8')
        self._fontconfig.FcPatternAddString(self._pattern, name, asbytes(value))

    def _set_double(self, name, value):
        assert self._pattern
        assert name
        assert self._fontconfig
        if not value:
            return
        self._fontconfig.FcPatternAddDouble(self._pattern, name, c_double(value))

    def _set_integer(self, name, value):
        assert self._pattern
        assert name
        assert self._fontconfig
        if not value:
            return
        self._fontconfig.FcPatternAddInteger(self._pattern, name, c_int(value))

    def _get_value(self, name):
        assert self._pattern
        assert name
        assert self._fontconfig
        value = FcValue()
        result = self._fontconfig.FcPatternGet(self._pattern, name, 0, byref(value))
        if _handle_fcresult(result):
            return value
        else:
            return None

    def _get_string(self, name):
        value = self._get_value(name)
        if value and value.type == FcTypeString:
            return asstr(value.u.s)
        else:
            return None

    def _get_face(self, name):
        value = self._get_value(name)
        if value and value.type == FcTypeFTFace:
            return value.u.f
        else:
            return None

    def _get_integer(self, name):
        value = self._get_value(name)
        if value and value.type == FcTypeInteger:
            return value.u.i
        else:
            return None

    def _get_double(self, name):
        value = self._get_value(name)
        if value and value.type == FcTypeDouble:
            return value.u.d
        else:
            return None