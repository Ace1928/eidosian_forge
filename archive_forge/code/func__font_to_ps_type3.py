import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
def _font_to_ps_type3(font_path, chars):
    """
    Subset *chars* from the font at *font_path* into a Type 3 font.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.

    Returns
    -------
    str
        The string representation of a Type 3 font, which can be included
        verbatim into a PostScript file.
    """
    font = get_font(font_path, hinting_factor=1)
    glyph_ids = [font.get_char_index(c) for c in chars]
    preamble = '%!PS-Adobe-3.0 Resource-Font\n%%Creator: Converted from TrueType to Type 3 by Matplotlib.\n10 dict begin\n/FontName /{font_name} def\n/PaintType 0 def\n/FontMatrix [{inv_units_per_em} 0 0 {inv_units_per_em} 0 0] def\n/FontBBox [{bbox}] def\n/FontType 3 def\n/Encoding [{encoding}] def\n/CharStrings {num_glyphs} dict dup begin\n/.notdef 0 def\n'.format(font_name=font.postscript_name, inv_units_per_em=1 / font.units_per_EM, bbox=' '.join(map(str, font.bbox)), encoding=' '.join((f'/{font.get_glyph_name(glyph_id)}' for glyph_id in glyph_ids)), num_glyphs=len(glyph_ids) + 1)
    postamble = '\nend readonly def\n\n/BuildGlyph {\n exch begin\n CharStrings exch\n 2 copy known not {pop /.notdef} if\n true 3 1 roll get exec\n end\n} _d\n\n/BuildChar {\n 1 index /Encoding get exch get\n 1 index /BuildGlyph get exec\n} _d\n\nFontName currentdict end definefont pop\n'
    entries = []
    for glyph_id in glyph_ids:
        g = font.load_glyph(glyph_id, LOAD_NO_SCALE)
        v, c = font.get_path()
        entries.append('/%(name)s{%(bbox)s sc\n' % {'name': font.get_glyph_name(glyph_id), 'bbox': ' '.join(map(str, [g.horiAdvance, 0, *g.bbox]))} + _path.convert_to_string(Path(v * 64, c), None, None, False, None, 0, [b'm', b'l', b'', b'c', b''], True).decode('ascii') + 'ce} _d')
    return preamble + '\n'.join(entries) + postamble