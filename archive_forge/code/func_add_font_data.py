import math
import warnings
from ctypes import c_void_p, c_int32, byref, c_byte
from pyglet.font import base
import pyglet.image
from pyglet.libs.darwin import cocoapy, kCTFontURLAttribute, CGFloat
@classmethod
def add_font_data(cls, data):
    dataRef = c_void_p(cf.CFDataCreate(None, data, len(data)))
    provider = c_void_p(quartz.CGDataProviderCreateWithCFData(dataRef))
    cgFont = c_void_p(quartz.CGFontCreateWithDataProvider(provider))
    cf.CFRelease(dataRef)
    quartz.CGDataProviderRelease(provider)
    ctFont = c_void_p(ct.CTFontCreateWithGraphicsFont(cgFont, 1, None, None))
    string = c_void_p(ct.CTFontCopyFamilyName(ctFont))
    familyName = str(cocoapy.cfstring_to_string(string))
    cf.CFRelease(string)
    string = c_void_p(ct.CTFontCopyFullName(ctFont))
    fullName = str(cocoapy.cfstring_to_string(string))
    cf.CFRelease(string)
    traits = ct.CTFontGetSymbolicTraits(ctFont)
    cf.CFRelease(ctFont)
    if familyName not in cls._loaded_CGFont_table:
        cls._loaded_CGFont_table[familyName] = {}
    cls._loaded_CGFont_table[familyName][traits] = cgFont
    if fullName not in cls._loaded_CGFont_table:
        cls._loaded_CGFont_table[fullName] = {}
    cls._loaded_CGFont_table[fullName][traits] = cgFont