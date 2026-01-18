from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class TeX(MathText):
    """ Render mathematical content using `LaTeX <https://www.latex-project.org/>`_
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
    inline = Bool(default=False, help='\n    Whether the math text is inline display or not (for TeX input). Default is False.\n    ')