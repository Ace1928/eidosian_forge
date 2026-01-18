from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
class GdkPixbuf2ImageDecoder(ImageDecoder):

    def get_file_extensions(self):
        return ['.png', '.xpm', '.jpg', '.jpeg', '.tif', '.tiff', '.pnm', '.ras', '.bmp', '.gif']

    def get_animation_file_extensions(self):
        return ['.gif', '.ani']

    def decode(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        loader = GdkPixBufLoader(filename, file)
        return loader.get_pixbuf().to_image()

    def decode_animation(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        loader = GdkPixBufLoader(filename, file)
        return loader.get_animation().to_animation()