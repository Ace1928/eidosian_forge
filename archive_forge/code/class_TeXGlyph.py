from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class TeXGlyph(MathTextGlyph):
    """
    Render mathematical content using `LaTeX <https://www.latex-project.org/>`_
    notation.

    See :ref:`ug_styling_mathtext` in the |user guide| for more information.

    .. note::
        Bokeh uses `MathJax <https://www.mathjax.org>`_ to render text
        containing mathematical notation.

        MathJax only supports math-mode macros (no text-mode macros). You
        can see more about differences between standard TeX/LaTeX and MathJax
        here: https://docs.mathjax.org/en/latest/input/tex/differences.html

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    macros = Dict(String, Either(String, Tuple(String, Int)), help='\n    User defined TeX macros.\n\n    This is a mapping from control sequence names (without leading backslash) to\n    either replacement strings or tuples of a replacement string and a number\n    of arguments.\n\n    Example:\n\n    .. code-block:: python\n\n        TeX(text=r"\\R \\rightarrow \\R^2", macros={"RR": r"{\\bf R}"})\n\n    ')
    display = Either(Enum('inline', 'block', 'auto'), default='auto', help='\n    Defines how the text is interpreted and what TeX display mode to use.\n\n    The following values are allowed:\n\n    * ``"auto"`` (the default)\n      The text is parsed, requiring TeX delimiters to enclose math content,\n      e.g. ``"$$x^2$$"`` or ``r"\\[\\frac{x}{y}\\]"``. This allows mixed\n      math text and regular text content. TeX display mode is inferred by\n      the parser.\n    * ``"block"``\n      The text is taken verbatim and TeX\'s block mode is used.\n    * ``"inline"``\n      The text is taken verbatim and TeX\'s inline mode is used.\n    ')