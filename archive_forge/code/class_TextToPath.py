from collections import OrderedDict
import logging
import urllib.parse
import numpy as np
from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import (
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_TARGET_LIGHT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
class TextToPath:
    """A class that converts strings to paths."""
    FONT_SCALE = 100.0
    DPI = 72

    def __init__(self):
        self.mathtext_parser = MathTextParser('path')
        self._texmanager = None

    def _get_font(self, prop):
        """
        Find the `FT2Font` matching font properties *prop*, with its size set.
        """
        filenames = _fontManager._find_fonts_by_props(prop)
        font = get_font(filenames)
        font.set_size(self.FONT_SCALE, self.DPI)
        return font

    def _get_hinting_flag(self):
        return LOAD_NO_HINTING

    def _get_char_id(self, font, ccode):
        """
        Return a unique id for the given font and character-code set.
        """
        return urllib.parse.quote(f'{font.postscript_name}-{ccode:x}')

    def get_text_width_height_descent(self, s, prop, ismath):
        fontsize = prop.get_size_in_points()
        if ismath == 'TeX':
            return TexManager().get_text_width_height_descent(s, fontsize)
        scale = fontsize / self.FONT_SCALE
        if ismath:
            prop = prop.copy()
            prop.set_size(self.FONT_SCALE)
            width, height, descent, *_ = self.mathtext_parser.parse(s, 72, prop)
            return (width * scale, height * scale, descent * scale)
        font = self._get_font(prop)
        font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
        w, h = font.get_width_height()
        w /= 64.0
        h /= 64.0
        d = font.get_descent()
        d /= 64.0
        return (w * scale, h * scale, d * scale)

    def get_text_path(self, prop, s, ismath=False):
        """
        Convert text *s* to path (a tuple of vertices and codes for
        matplotlib.path.Path).

        Parameters
        ----------
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties for the text.
        s : str
            The text to be converted.
        ismath : {False, True, "TeX"}
            If True, use mathtext parser.  If "TeX", use tex for rendering.

        Returns
        -------
        verts : list
            A list of arrays containing the (x, y) coordinates of the vertices.
        codes : list
            A list of path codes.

        Examples
        --------
        Create a list of vertices and codes from a text, and create a `.Path`
        from those::

            from matplotlib.path import Path
            from matplotlib.text import TextToPath
            from matplotlib.font_manager import FontProperties

            fp = FontProperties(family="Comic Neue", style="italic")
            verts, codes = TextToPath().get_text_path(fp, "ABC")
            path = Path(verts, codes, closed=False)

        Also see `TextPath` for a more direct way to create a path from a text.
        """
        if ismath == 'TeX':
            glyph_info, glyph_map, rects = self.get_glyphs_tex(prop, s)
        elif not ismath:
            font = self._get_font(prop)
            glyph_info, glyph_map, rects = self.get_glyphs_with_font(font, s)
        else:
            glyph_info, glyph_map, rects = self.get_glyphs_mathtext(prop, s)
        verts, codes = ([], [])
        for glyph_id, xposition, yposition, scale in glyph_info:
            verts1, codes1 = glyph_map[glyph_id]
            verts.extend(verts1 * scale + [xposition, yposition])
            codes.extend(codes1)
        for verts1, codes1 in rects:
            verts.extend(verts1)
            codes.extend(codes1)
        if not verts:
            verts = np.empty((0, 2))
        return (verts, codes)

    def get_glyphs_with_font(self, font, s, glyph_map=None, return_new_glyphs_only=False):
        """
        Convert string *s* to vertices and codes using the provided ttf font.
        """
        if glyph_map is None:
            glyph_map = OrderedDict()
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map
        xpositions = []
        glyph_ids = []
        for item in _text_helpers.layout(s, font):
            char_id = self._get_char_id(item.ft_object, ord(item.char))
            glyph_ids.append(char_id)
            xpositions.append(item.x)
            if char_id not in glyph_map:
                glyph_map_new[char_id] = item.ft_object.get_path()
        ypositions = [0] * len(xpositions)
        sizes = [1.0] * len(xpositions)
        rects = []
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, rects)

    def get_glyphs_mathtext(self, prop, s, glyph_map=None, return_new_glyphs_only=False):
        """
        Parse mathtext string *s* and convert it to a (vertices, codes) pair.
        """
        prop = prop.copy()
        prop.set_size(self.FONT_SCALE)
        width, height, descent, glyphs, rects = self.mathtext_parser.parse(s, self.DPI, prop)
        if not glyph_map:
            glyph_map = OrderedDict()
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map
        xpositions = []
        ypositions = []
        glyph_ids = []
        sizes = []
        for font, fontsize, ccode, ox, oy in glyphs:
            char_id = self._get_char_id(font, ccode)
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                font.load_char(ccode, flags=LOAD_NO_HINTING)
                glyph_map_new[char_id] = font.get_path()
            xpositions.append(ox)
            ypositions.append(oy)
            glyph_ids.append(char_id)
            size = fontsize / self.FONT_SCALE
            sizes.append(size)
        myrects = []
        for ox, oy, w, h in rects:
            vert1 = [(ox, oy), (ox, oy + h), (ox + w, oy + h), (ox + w, oy), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            myrects.append((vert1, code1))
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, myrects)

    def get_glyphs_tex(self, prop, s, glyph_map=None, return_new_glyphs_only=False):
        """Convert the string *s* to vertices and codes using usetex mode."""
        dvifile = TexManager().make_dvi(s, self.FONT_SCALE)
        with dviread.Dvi(dvifile, self.DPI) as dvi:
            page, = dvi
        if glyph_map is None:
            glyph_map = OrderedDict()
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map
        glyph_ids, xpositions, ypositions, sizes = ([], [], [], [])
        for text in page.text:
            font = get_font(text.font_path)
            char_id = self._get_char_id(font, text.glyph)
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                glyph_name_or_index = text.glyph_name_or_index
                if isinstance(glyph_name_or_index, str):
                    index = font.get_name_index(glyph_name_or_index)
                    font.load_glyph(index, flags=LOAD_TARGET_LIGHT)
                elif isinstance(glyph_name_or_index, int):
                    self._select_native_charmap(font)
                    font.load_char(glyph_name_or_index, flags=LOAD_TARGET_LIGHT)
                else:
                    raise TypeError(f'Glyph spec of unexpected type: {glyph_name_or_index!r}')
                glyph_map_new[char_id] = font.get_path()
            glyph_ids.append(char_id)
            xpositions.append(text.x)
            ypositions.append(text.y)
            sizes.append(text.font_size / self.FONT_SCALE)
        myrects = []
        for ox, oy, h, w in page.boxes:
            vert1 = [(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            myrects.append((vert1, code1))
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)), glyph_map_new, myrects)

    @staticmethod
    def _select_native_charmap(font):
        for charmap_code in [1094992451, 1094995778]:
            try:
                font.select_charmap(charmap_code)
            except (ValueError, RuntimeError):
                pass
            else:
                break
        else:
            _log.warning('No supported encoding in font (%s).', font.fname)