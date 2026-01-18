from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
class FontConfigSearchPattern(FontConfigPattern):

    def __init__(self, fontconfig):
        super(FontConfigSearchPattern, self).__init__(fontconfig)
        self.name = None
        self.bold = False
        self.italic = False
        self.size = None

    def match(self):
        self._prepare_search_pattern()
        result_pattern = self._get_match()
        if result_pattern:
            return FontConfigSearchResult(self._fontconfig, result_pattern)
        else:
            return None

    def _prepare_search_pattern(self):
        self._create()
        self._set_string(FC_FAMILY, self.name)
        self._set_double(FC_SIZE, self.size)
        self._set_integer(FC_WEIGHT, self._bold_to_weight(self.bold))
        self._set_integer(FC_SLANT, self._italic_to_slant(self.italic))
        self._substitute_defaults()

    def _substitute_defaults(self):
        assert self._pattern
        assert self._fontconfig
        self._fontconfig.FcConfigSubstitute(None, self._pattern, FcMatchPattern)
        self._fontconfig.FcDefaultSubstitute(self._pattern)

    def _get_match(self):
        assert self._pattern
        assert self._fontconfig
        match_result = FcResult()
        match_pattern = self._fontconfig.FcFontMatch(0, self._pattern, byref(match_result))
        if _handle_fcresult(match_result.value):
            return match_pattern
        else:
            return None

    def dispose(self):
        self._destroy()