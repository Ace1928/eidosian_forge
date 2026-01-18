from ctypes import c_void_p, c_ubyte
from pyglet.image import ImageData, Animation, AnimationFrame
from pyglet.image.codecs import *
from pyglet.libs.darwin.cocoapy import cf, quartz, NSMakeRect
from pyglet.libs.darwin.cocoapy import cfnumber_to_number
from pyglet.libs.darwin.cocoapy import kCGImageAlphaPremultipliedLast
from pyglet.libs.darwin.cocoapy import kCGImagePropertyGIFDictionary
from pyglet.libs.darwin.cocoapy import kCGImagePropertyGIFDelayTime
def _get_pyglet_ImageData_from_source_at_index(self, sourceRef, index):
    imageRef = c_void_p(quartz.CGImageSourceCreateImageAtIndex(sourceRef, index, None))
    format = 'RGBA'
    rgbColorSpace = c_void_p(quartz.CGColorSpaceCreateDeviceRGB())
    bitsPerComponent = 8
    width = quartz.CGImageGetWidth(imageRef)
    height = quartz.CGImageGetHeight(imageRef)
    bytesPerRow = 4 * width
    bufferSize = height * bytesPerRow
    buffer = (c_ubyte * bufferSize)()
    bitmap = c_void_p(quartz.CGBitmapContextCreate(buffer, width, height, bitsPerComponent, bytesPerRow, rgbColorSpace, kCGImageAlphaPremultipliedLast))
    quartz.CGContextDrawImage(bitmap, NSMakeRect(0, 0, width, height), imageRef)
    quartz.CGImageRelease(imageRef)
    quartz.CGContextRelease(bitmap)
    quartz.CGColorSpaceRelease(rgbColorSpace)
    pitch = bytesPerRow
    return ImageData(width, height, format, buffer, -pitch)