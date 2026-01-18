import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def _get_texture_atlas_bin(self, width, height, border):
    """A heuristic for determining the atlas bin to use for a given image
        size.  Returns None if the image should not be placed in an atlas (too
        big), otherwise the bin (a list of TextureAtlas).
        """
    max_texture_size = pyglet.image.get_max_texture_size()
    max_size = min(2048, max_texture_size) - border
    if width > max_size or height > max_size:
        return None
    bin_size = 1
    if height > max_size / 4:
        bin_size = 2
    try:
        texture_bin = self._texture_atlas_bins[bin_size]
    except KeyError:
        texture_bin = pyglet.image.atlas.TextureBin()
        self._texture_atlas_bins[bin_size] = texture_bin
    return texture_bin