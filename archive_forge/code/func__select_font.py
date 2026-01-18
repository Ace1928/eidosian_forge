from PIL import Image, ImageFont, ImageDraw
from kivy.compat import text_type
from kivy.core.text import LabelBase
from kivy.core.image import ImageData
def _select_font(self):
    if self.options['font_size'] < 1:
        return None
    fontsize = int(self.options['font_size'])
    fontname = self.options['font_name_r']
    try:
        id = '%s.%s' % (text_type(fontname), text_type(fontsize))
    except UnicodeDecodeError:
        id = '%s.%s' % (fontname, fontsize)
    if id not in self._cache:
        font = ImageFont.truetype(fontname, fontsize)
        self._cache[id] = font
    return self._cache[id]