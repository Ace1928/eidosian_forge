import unicodedata
from pyglet.gl import *
from pyglet import image
class GlyphTextureBin(image.atlas.TextureBin):
    """Same as a TextureBin but allows you to specify filter of Glyphs."""

    def add(self, img, fmt=GL_RGBA, min_filter=GL_LINEAR, mag_filter=GL_LINEAR, border=0):
        for atlas in list(self.atlases):
            try:
                return atlas.add(img, border)
            except image.atlas.AllocatorException:
                if img.width < 64 and img.height < 64:
                    self.atlases.remove(atlas)
        atlas = GlyphTextureAtlas(self.texture_width, self.texture_height, fmt, min_filter, mag_filter)
        self.atlases.append(atlas)
        return atlas.add(img, border)