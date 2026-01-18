from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
@staticmethod
def _load_fontconfig_library():
    fontconfig = pyglet.lib.load_library('fontconfig')
    fontconfig.FcInit()
    fontconfig.FcPatternBuild.restype = c_void_p
    fontconfig.FcPatternCreate.restype = c_void_p
    fontconfig.FcFontMatch.restype = c_void_p
    fontconfig.FcFreeTypeCharIndex.restype = c_uint
    fontconfig.FcPatternAddDouble.argtypes = [c_void_p, c_char_p, c_double]
    fontconfig.FcPatternAddInteger.argtypes = [c_void_p, c_char_p, c_int]
    fontconfig.FcPatternAddString.argtypes = [c_void_p, c_char_p, c_char_p]
    fontconfig.FcConfigSubstitute.argtypes = [c_void_p, c_void_p, c_int]
    fontconfig.FcDefaultSubstitute.argtypes = [c_void_p]
    fontconfig.FcFontMatch.argtypes = [c_void_p, c_void_p, c_void_p]
    fontconfig.FcPatternDestroy.argtypes = [c_void_p]
    fontconfig.FcPatternGetFTFace.argtypes = [c_void_p, c_char_p, c_int, c_void_p]
    fontconfig.FcPatternGet.argtypes = [c_void_p, c_char_p, c_int, c_void_p]
    return fontconfig