import os.path
from pyglet.image import *
from pyglet.image.codecs import *
from PIL import Image
class PILImageEncoder(ImageEncoder):

    def get_file_extensions(self):
        return ['.bmp', '.eps', '.gif', '.jpg', '.jpeg', '.pcx', '.png', '.ppm', '.tiff', '.xbm']

    def encode(self, image, filename, file):
        pil_format = filename and os.path.splitext(filename)[1][1:] or 'png'
        if pil_format.lower() == 'jpg':
            pil_format = 'JPEG'
        image = image.get_image_data()
        fmt = image.format
        if fmt != 'RGB':
            fmt = 'RGBA'
        pitch = -(image.width * len(fmt))
        try:
            image_from_fn = getattr(Image, 'frombytes')
        except AttributeError:
            image_from_fn = getattr(Image, 'fromstring')
        pil_image = image_from_fn(fmt, (image.width, image.height), image.get_data(fmt, pitch))
        try:
            pil_image.save(file, pil_format)
        except Exception as e:
            raise ImageEncodeException(e)