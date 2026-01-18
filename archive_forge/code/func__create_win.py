import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _create_win(self):
    try:
        key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, 'Software\\Microsoft\\Windows NT\\CurrentVersion\\Fonts')
    except EnvironmentError:
        try:
            key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, 'Software\\Microsoft\\Windows\\CurrentVersion\\Fonts')
        except EnvironmentError:
            raise FontNotFound("Can't open Windows font registry key")
    try:
        path = self._lookup_win(key, self.font_name, STYLES['NORMAL'], True)
        self.fonts['NORMAL'] = ImageFont.truetype(path, self.font_size)
        for style in ('ITALIC', 'BOLD', 'BOLDITALIC'):
            path = self._lookup_win(key, self.font_name, STYLES[style])
            if path:
                self.fonts[style] = ImageFont.truetype(path, self.font_size)
            elif style == 'BOLDITALIC':
                self.fonts[style] = self.fonts['BOLD']
            else:
                self.fonts[style] = self.fonts['NORMAL']
    finally:
        _winreg.CloseKey(key)