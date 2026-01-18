import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _create_mac(self):
    font_map = {}
    for font_dir in (os.path.join(os.getenv('HOME'), 'Library/Fonts/'), '/Library/Fonts/', '/System/Library/Fonts/'):
        font_map.update(((os.path.splitext(f)[0].lower(), os.path.join(font_dir, f)) for f in os.listdir(font_dir) if f.lower().endswith('ttf')))
    for name in STYLES['NORMAL']:
        path = self._get_mac_font_path(font_map, self.font_name, name)
        if path is not None:
            self.fonts['NORMAL'] = ImageFont.truetype(path, self.font_size)
            break
    else:
        raise FontNotFound('No usable fonts named: "%s"' % self.font_name)
    for style in ('ITALIC', 'BOLD', 'BOLDITALIC'):
        for stylename in STYLES[style]:
            path = self._get_mac_font_path(font_map, self.font_name, stylename)
            if path is not None:
                self.fonts[style] = ImageFont.truetype(path, self.font_size)
                break
        else:
            if style == 'BOLDITALIC':
                self.fonts[style] = self.fonts['BOLD']
            else:
                self.fonts[style] = self.fonts['NORMAL']