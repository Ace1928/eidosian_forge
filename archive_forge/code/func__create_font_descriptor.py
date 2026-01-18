import math
import warnings
from ctypes import c_void_p, c_int32, byref, c_byte
from pyglet.font import base
import pyglet.image
from pyglet.libs.darwin import cocoapy, kCTFontURLAttribute, CGFloat
def _create_font_descriptor(self, family_name, traits):
    attributes = c_void_p(cf.CFDictionaryCreateMutable(None, 0, cf.kCFTypeDictionaryKeyCallBacks, cf.kCFTypeDictionaryValueCallBacks))
    cfname = cocoapy.CFSTR(family_name)
    cf.CFDictionaryAddValue(attributes, cocoapy.kCTFontFamilyNameAttribute, cfname)
    cf.CFRelease(cfname)
    itraits = c_int32(traits)
    symTraits = c_void_p(cf.CFNumberCreate(None, cocoapy.kCFNumberSInt32Type, byref(itraits)))
    if symTraits:
        traitsDict = c_void_p(cf.CFDictionaryCreateMutable(None, 0, cf.kCFTypeDictionaryKeyCallBacks, cf.kCFTypeDictionaryValueCallBacks))
        if traitsDict:
            cf.CFDictionaryAddValue(traitsDict, cocoapy.kCTFontSymbolicTrait, symTraits)
            cf.CFDictionaryAddValue(attributes, cocoapy.kCTFontTraitsAttribute, traitsDict)
            cf.CFRelease(traitsDict)
        cf.CFRelease(symTraits)
    descriptor = c_void_p(ct.CTFontDescriptorCreateWithAttributes(attributes))
    cf.CFRelease(attributes)
    return descriptor