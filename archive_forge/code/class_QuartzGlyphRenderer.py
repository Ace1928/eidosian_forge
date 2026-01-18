import math
import warnings
from ctypes import c_void_p, c_int32, byref, c_byte
from pyglet.font import base
import pyglet.image
from pyglet.libs.darwin import cocoapy, kCTFontURLAttribute, CGFloat
class QuartzGlyphRenderer(base.GlyphRenderer):

    def __init__(self, font):
        super().__init__(font)
        self.font = font

    def render(self, text):
        ctFont = self.font.ctFont
        attributes = c_void_p(cf.CFDictionaryCreateMutable(None, 1, cf.kCFTypeDictionaryKeyCallBacks, cf.kCFTypeDictionaryValueCallBacks))
        cf.CFDictionaryAddValue(attributes, cocoapy.kCTFontAttributeName, ctFont)
        string = c_void_p(cf.CFAttributedStringCreate(None, cocoapy.CFSTR(text), attributes))
        line = c_void_p(ct.CTLineCreateWithAttributedString(string))
        cf.CFRelease(string)
        cf.CFRelease(attributes)
        count = len(text)
        chars = (cocoapy.UniChar * count)(*list(map(ord, str(text))))
        glyphs = (cocoapy.CGGlyph * count)()
        ct.CTFontGetGlyphsForCharacters(ctFont, chars, glyphs, count)
        if glyphs[0] == 0:
            ascent, descent = (CGFloat(), CGFloat())
            advance = width = int(ct.CTLineGetTypographicBounds(line, byref(ascent), byref(descent), None))
            height = int(ascent.value + descent.value)
            lsb = 0
            baseline = descent.value
        else:
            rect = ct.CTFontGetBoundingRectsForGlyphs(ctFont, 0, glyphs, None, count)
            advance = ct.CTFontGetAdvancesForGlyphs(ctFont, 0, glyphs, None, count)
            width = max(int(math.ceil(rect.size.width) + 2), 1)
            height = max(int(math.ceil(rect.size.height) + 2), 1)
            baseline = -int(math.floor(rect.origin.y)) + 1
            lsb = int(math.ceil(rect.origin.x)) - 1
            advance = int(round(advance))
        bitsPerComponent = 8
        bytesPerRow = 4 * width
        colorSpace = c_void_p(quartz.CGColorSpaceCreateDeviceRGB())
        bitmap = c_void_p(quartz.CGBitmapContextCreate(None, width, height, bitsPerComponent, bytesPerRow, colorSpace, cocoapy.kCGImageAlphaPremultipliedLast))
        quartz.CGContextSetShouldAntialias(bitmap, True)
        quartz.CGContextSetTextPosition(bitmap, -lsb, baseline)
        ct.CTLineDraw(line, bitmap)
        cf.CFRelease(line)
        imageRef = c_void_p(quartz.CGBitmapContextCreateImage(bitmap))
        bytesPerRow = quartz.CGImageGetBytesPerRow(imageRef)
        dataProvider = c_void_p(quartz.CGImageGetDataProvider(imageRef))
        imageData = c_void_p(quartz.CGDataProviderCopyData(dataProvider))
        buffersize = cf.CFDataGetLength(imageData)
        buffer = (c_byte * buffersize)()
        byteRange = cocoapy.CFRange(0, buffersize)
        cf.CFDataGetBytes(imageData, byteRange, buffer)
        quartz.CGImageRelease(imageRef)
        quartz.CGDataProviderRelease(imageData)
        cf.CFRelease(bitmap)
        cf.CFRelease(colorSpace)
        glyph_image = pyglet.image.ImageData(width, height, 'RGBA', buffer, bytesPerRow)
        glyph = self.font.create_glyph(glyph_image)
        glyph.set_bearings(baseline, lsb, advance)
        t = list(glyph.tex_coords)
        glyph.tex_coords = t[9:12] + t[6:9] + t[3:6] + t[:3]
        return glyph