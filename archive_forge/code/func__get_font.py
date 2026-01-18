from kivy.compat import PY2
from kivy.core.text import LabelBase
from kivy.core.image import ImageData
from kivy.utils import deprecated
def _get_font(self):
    fontid = self._get_font_id()
    if fontid not in pygame_cache:
        font_handle = fontobject = None
        fontname = self.options['font_name_r']
        ext = fontname.rsplit('.', 1)
        if len(ext) == 2:
            font_handle = open(fontname, 'rb')
            fontobject = pygame.font.Font(font_handle, int(self.options['font_size']))
        if fontobject is None:
            font = pygame.font.match_font(self.options['font_name_r'].replace(' ', ''), bold=self.options['bold'], italic=self.options['italic'])
            fontobject = pygame.font.Font(font, int(self.options['font_size']))
        pygame_cache[fontid] = fontobject
        pygame_font_handles[fontid] = font_handle
        pygame_cache_order.append(fontid)
    while len(pygame_cache_order) > 64:
        popid = pygame_cache_order.pop(0)
        del pygame_cache[popid]
        font_handle = pygame_font_handles.pop(popid)
        if font_handle is not None:
            font_handle.close()
    return pygame_cache[fontid]