from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class WICEncoder(ImageEncoder):

    def get_file_extensions(self):
        return [ext for ext in extension_to_container]

    def encode(self, image, filename, file):
        image = image.get_image_data()
        wicstream = IWICStream()
        encoder = IWICBitmapEncoder()
        frame = IWICBitmapFrameEncode()
        property_bag = IPropertyBag2()
        ext = filename and os.path.splitext(filename)[1] or '.png'
        container = extension_to_container.get(ext, GUID_ContainerFormatPng)
        _factory.CreateStream(byref(wicstream))
        if container == GUID_ContainerFormatJpeg:
            fmt = 'BGR'
            default_format = GUID_WICPixelFormat24bppBGR
        elif len(image.format) == 3:
            fmt = 'BGR'
            default_format = GUID_WICPixelFormat24bppBGR
        else:
            fmt = 'BGRA'
            default_format = GUID_WICPixelFormat32bppBGRA
        pitch = image.width * len(fmt)
        image_data = image.get_data(fmt, -pitch)
        size = pitch * image.height
        if file:
            istream = IStream()
            ole32.CreateStreamOnHGlobal(None, True, byref(istream))
            wicstream.InitializeFromIStream(istream)
        else:
            wicstream.InitializeFromFilename(filename, GENERIC_WRITE)
        _factory.CreateEncoder(container, None, byref(encoder))
        encoder.Initialize(wicstream, WICBitmapEncoderNoCache)
        encoder.CreateNewFrame(byref(frame), byref(property_bag))
        frame.Initialize(property_bag)
        frame.SetSize(image.width, image.height)
        frame.SetPixelFormat(byref(default_format))
        data = (BYTE * size).from_buffer(bytearray(image_data))
        frame.WritePixels(image.height, pitch, size, data)
        frame.Commit()
        encoder.Commit()
        if file:
            sts = STATSTG()
            istream.Stat(byref(sts), 0)
            stream_size = sts.cbSize
            istream.Seek(0, STREAM_SEEK_SET, None)
            buf = (BYTE * stream_size)()
            written = ULONG()
            istream.Read(byref(buf), stream_size, byref(written))
            if written.value == stream_size:
                file.write(buf)
            else:
                print(f'Failed to read all of the data from stream attempting to save {file}')
            istream.Release()
        encoder.Release()
        frame.Release()
        property_bag.Release()
        wicstream.Release()