import functools
import logging
import matplotlib as mpl
from matplotlib import _api, _mathtext
from matplotlib.ft2font import LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties
from ._mathtext import (  # noqa: reexported API
class MathTextParser:
    _parser = None
    _font_type_mapping = {'cm': _mathtext.BakomaFonts, 'dejavuserif': _mathtext.DejaVuSerifFonts, 'dejavusans': _mathtext.DejaVuSansFonts, 'stix': _mathtext.StixFonts, 'stixsans': _mathtext.StixSansFonts, 'custom': _mathtext.UnicodeFonts}

    def __init__(self, output):
        """
        Create a MathTextParser for the given backend *output*.

        Parameters
        ----------
        output : {"path", "agg"}
            Whether to return a `VectorParse` ("path") or a
            `RasterParse` ("agg", or its synonym "macosx").
        """
        self._output_type = _api.check_getitem({'path': 'vector', 'agg': 'raster', 'macosx': 'raster'}, output=output.lower())

    def parse(self, s, dpi=72, prop=None, *, antialiased=None):
        """
        Parse the given math expression *s* at the given *dpi*.  If *prop* is
        provided, it is a `.FontProperties` object specifying the "default"
        font to use in the math expression, used for all non-math text.

        The results are cached, so multiple calls to `parse`
        with the same expression should be fast.

        Depending on the *output* type, this returns either a `VectorParse` or
        a `RasterParse`.
        """
        prop = prop.copy() if prop is not None else None
        antialiased = mpl._val_or_rc(antialiased, 'text.antialiased')
        return self._parse_cached(s, dpi, prop, antialiased)

    @functools.lru_cache(50)
    def _parse_cached(self, s, dpi, prop, antialiased):
        from matplotlib.backends import backend_agg
        if prop is None:
            prop = FontProperties()
        fontset_class = _api.check_getitem(self._font_type_mapping, fontset=prop.get_math_fontfamily())
        load_glyph_flags = {'vector': LOAD_NO_HINTING, 'raster': backend_agg.get_hinting_flag()}[self._output_type]
        fontset = fontset_class(prop, load_glyph_flags)
        fontsize = prop.get_size_in_points()
        if self._parser is None:
            self.__class__._parser = _mathtext.Parser()
        box = self._parser.parse(s, fontset, fontsize, dpi)
        output = _mathtext.ship(box)
        if self._output_type == 'vector':
            return output.to_vector()
        elif self._output_type == 'raster':
            return output.to_raster(antialiased=antialiased)