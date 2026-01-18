import unicodedata
from pyglet.gl import *
from pyglet import image
def _get_optimal_atlas_size(self, image_data):
    """Return the smallest size of atlas that can fit around 100 glyphs based on the image_data provided."""
    aw, ah = (self.texture_width, self.texture_height)
    atlas_size = None
    i = 0
    while not atlas_size:
        fit = ((aw - (image_data.width + 2)) // (image_data.width + 2) + 1) * ((ah - (image_data.height + 2)) // (image_data.height + 2) + 1)
        if fit >= self.glyph_fit:
            atlas_size = (aw, ah)
        if i % 2:
            aw *= 2
        else:
            ah *= 2
        i += 1
    return atlas_size