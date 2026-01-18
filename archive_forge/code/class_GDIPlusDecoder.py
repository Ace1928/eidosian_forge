from pyglet.libs.win32.com import pIUnknown
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
class GDIPlusDecoder(ImageDecoder):

    def get_file_extensions(self):
        return ['.bmp', '.gif', '.jpg', '.jpeg', '.exif', '.png', '.tif', '.tiff']

    def get_animation_file_extensions(self):
        return ['.gif']

    def _load_bitmap(self, filename, file):
        data = file.read()
        hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        ptr = kernel32.GlobalLock(hglob)
        memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(hglob)
        self.stream = pIUnknown()
        ole32.CreateStreamOnHGlobal(hglob, True, byref(self.stream))
        bitmap = c_void_p()
        status = gdiplus.GdipCreateBitmapFromStream(self.stream, byref(bitmap))
        if status != 0:
            self.stream.Release()
            raise ImageDecodeException('GDI+ cannot load %r' % (filename or file))
        return bitmap

    @staticmethod
    def _get_image(bitmap):
        width = REAL()
        height = REAL()
        gdiplus.GdipGetImageDimension(bitmap, byref(width), byref(height))
        width = int(width.value)
        height = int(height.value)
        pf = c_int()
        gdiplus.GdipGetImagePixelFormat(bitmap, byref(pf))
        pf = pf.value
        fmt = 'BGRA'
        if pf == PixelFormat24bppRGB:
            fmt = 'BGR'
        elif pf == PixelFormat32bppRGB:
            pass
        elif pf == PixelFormat32bppARGB:
            pass
        elif pf in (PixelFormat16bppARGB1555, PixelFormat32bppPARGB, PixelFormat64bppARGB, PixelFormat64bppPARGB):
            pf = PixelFormat32bppARGB
        else:
            fmt = 'BGR'
            pf = PixelFormat24bppRGB
        rect = Rect()
        rect.X = 0
        rect.Y = 0
        rect.Width = width
        rect.Height = height
        bitmap_data = BitmapData()
        gdiplus.GdipBitmapLockBits(bitmap, byref(rect), ImageLockModeRead, pf, byref(bitmap_data))
        buffer = create_string_buffer(bitmap_data.Stride * height)
        memmove(buffer, bitmap_data.Scan0, len(buffer))
        gdiplus.GdipBitmapUnlockBits(bitmap, byref(bitmap_data))
        return ImageData(width, height, fmt, buffer, -bitmap_data.Stride)

    def _delete_bitmap(self, bitmap):
        gdiplus.GdipDisposeImage(bitmap)
        self.stream.Release()

    def decode(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        bitmap = self._load_bitmap(filename, file)
        image = self._get_image(bitmap)
        self._delete_bitmap(bitmap)
        return image

    def decode_animation(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        bitmap = self._load_bitmap(filename, file)
        dimension_count = c_uint()
        gdiplus.GdipImageGetFrameDimensionsCount(bitmap, byref(dimension_count))
        if dimension_count.value < 1:
            self._delete_bitmap(bitmap)
            raise ImageDecodeException('Image has no frame dimensions')
        dimensions = (c_void_p * dimension_count.value)()
        gdiplus.GdipImageGetFrameDimensionsList(bitmap, dimensions, dimension_count.value)
        frame_count = c_uint()
        gdiplus.GdipImageGetFrameCount(bitmap, dimensions, byref(frame_count))
        prop_id = PropertyTagFrameDelay
        prop_size = c_uint()
        gdiplus.GdipGetPropertyItemSize(bitmap, prop_id, byref(prop_size))
        prop_buffer = c_buffer(prop_size.value)
        prop_item = cast(prop_buffer, POINTER(PropertyItem)).contents
        gdiplus.GdipGetPropertyItem(bitmap, prop_id, prop_size.value, prop_buffer)
        n_delays = prop_item.length // sizeof(c_long)
        delays = cast(prop_item.value, POINTER(c_long * n_delays)).contents
        frames = []
        for i in range(frame_count.value):
            gdiplus.GdipImageSelectActiveFrame(bitmap, dimensions, i)
            image = self._get_image(bitmap)
            delay = delays[i]
            if delay <= 1:
                delay = 10
            frames.append(AnimationFrame(image, delay / 100.0))
        self._delete_bitmap(bitmap)
        return Animation(frames)