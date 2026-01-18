import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def _alloc_image(self, name, atlas, border):
    file = self.file(name)
    try:
        img = pyglet.image.load(name, file=file)
    finally:
        file.close()
    if not atlas:
        return img.get_texture()
    bin = self._get_texture_atlas_bin(img.width, img.height, border)
    if bin is None:
        return img.get_texture()
    return bin.add(img, border)