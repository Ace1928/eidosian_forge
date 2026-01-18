import math
import warnings
from ctypes import c_void_p, c_int32, byref, c_byte
from pyglet.font import base
import pyglet.image
from pyglet.libs.darwin import cocoapy, kCTFontURLAttribute, CGFloat
class QuartzFont(base.Font):
    glyph_renderer_class = QuartzGlyphRenderer
    _loaded_CGFont_table = {}

    def _lookup_font_with_family_and_traits(self, family, traits):
        if family not in self._loaded_CGFont_table:
            return None
        fonts = self._loaded_CGFont_table[family]
        if not fonts:
            return None
        if traits in fonts:
            return fonts[traits]
        for t, f in fonts.items():
            if traits & t:
                return f
        if 0 in fonts:
            return fonts[0]
        return list(fonts.values())[0]

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

    def __init__(self, name, size, bold=False, italic=False, stretch=False, dpi=None):
        if stretch:
            warnings.warn('The current font render does not support stretching.')
        super().__init__()
        name = name or 'Helvetica'
        dpi = dpi or 96
        size = size * dpi / 72.0
        traits = 0
        if bold:
            traits |= cocoapy.kCTFontBoldTrait
        if italic:
            traits |= cocoapy.kCTFontItalicTrait
        name = str(name)
        self.traits = traits
        cgFont = self._lookup_font_with_family_and_traits(name, traits)
        if cgFont:
            self.ctFont = c_void_p(ct.CTFontCreateWithGraphicsFont(cgFont, size, None, None))
        else:
            descriptor = self._create_font_descriptor(name, traits)
            self.ctFont = c_void_p(ct.CTFontCreateWithFontDescriptor(descriptor, size, None))
            cf.CFRelease(descriptor)
            assert self.ctFont, "Couldn't load font: " + name
        string = c_void_p(ct.CTFontCopyFamilyName(self.ctFont))
        self._family_name = str(cocoapy.cfstring_to_string(string))
        cf.CFRelease(string)
        self.ascent = int(math.ceil(ct.CTFontGetAscent(self.ctFont)))
        self.descent = -int(math.ceil(ct.CTFontGetDescent(self.ctFont)))

    @property
    def filename(self):
        descriptor = self._create_font_descriptor(self.name, self.traits)
        ref = c_void_p(ct.CTFontDescriptorCopyAttribute(descriptor, kCTFontURLAttribute))
        if ref:
            url = cocoapy.ObjCInstance(ref, cache=False)
            filepath = url.fileSystemRepresentation().decode()
            cf.CFRelease(ref)
            return filepath
        cf.CFRelease(descriptor)
        return 'Unknown'

    @property
    def name(self):
        return self._family_name

    def __del__(self):
        cf.CFRelease(self.ctFont)

    @classmethod
    def have_font(cls, name):
        name = str(name)
        if name in cls._loaded_CGFont_table:
            return True
        cfstring = cocoapy.CFSTR(name)
        cgfont = c_void_p(quartz.CGFontCreateWithFontName(cfstring))
        cf.CFRelease(cfstring)
        if cgfont:
            cf.CFRelease(cgfont)
            return True
        return False

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