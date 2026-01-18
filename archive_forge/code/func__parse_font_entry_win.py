import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def _parse_font_entry_win(name, font, fonts):
    """
    Parse out a simpler name and the font style from the initial file name.

    :param name: The font name
    :param font: The font file path
    :param fonts: The pygame font dictionary
    """
    true_type_suffix = '(TrueType)'
    mods = ('demibold', 'narrow', 'light', 'unicode', 'bt', 'mt')
    if name.endswith(true_type_suffix):
        name = name.rstrip(true_type_suffix).rstrip()
    name = name.lower().split()
    bold = italic = False
    for mod in mods:
        if mod in name:
            name.remove(mod)
    if 'bold' in name:
        name.remove('bold')
        bold = True
    if 'italic' in name:
        name.remove('italic')
        italic = True
    name = ''.join(name)
    name = _simplename(name)
    _addfont(name, bold, italic, font, fonts)