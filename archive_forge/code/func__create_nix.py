import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _create_nix(self):
    for name in STYLES['NORMAL']:
        path = self._get_nix_font_path(self.font_name, name)
        if path is not None:
            self.fonts['NORMAL'] = ImageFont.truetype(path, self.font_size)
            break
    else:
        raise FontNotFound('No usable fonts named: "%s"' % self.font_name)
    for style in ('ITALIC', 'BOLD', 'BOLDITALIC'):
        for stylename in STYLES[style]:
            path = self._get_nix_font_path(self.font_name, stylename)
            if path is not None:
                self.fonts[style] = ImageFont.truetype(path, self.font_size)
                break
        else:
            if style == 'BOLDITALIC':
                self.fonts[style] = self.fonts['BOLD']
            else:
                self.fonts[style] = self.fonts['NORMAL']