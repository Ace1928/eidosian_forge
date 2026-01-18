import array
import itertools
from pyglet.image import ImageData, ImageDecodeException
from pyglet.image.codecs import ImageDecoder, ImageEncoder
import pyglet.extlibs.png as pypng
class PNGImageEncoder(ImageEncoder):

    def get_file_extensions(self):
        return ['.png']

    def encode(self, image, filename, file):
        image = image.get_image_data()
        has_alpha = 'A' in image.format
        greyscale = len(image.format) < 3
        if has_alpha:
            if greyscale:
                image.format = 'LA'
            else:
                image.format = 'RGBA'
        elif greyscale:
            image.format = 'L'
        else:
            image.format = 'RGB'
        image.pitch = -(image.width * len(image.format))
        writer = pypng.Writer(image.width, image.height, greyscale=greyscale, alpha=has_alpha)
        data = array.array('B')
        data.frombytes(image.get_data(image.format, image.pitch))
        writer.write_array(file, data)