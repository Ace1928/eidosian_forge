from __future__ import annotations
import re
import sys
from typing import (
import param  # type: ignore
from pyviz_comms import Comm, JupyterComm  # type: ignore
from ..io.resources import CDN_DIST
from ..util import lazy_load
from .base import ModelPane

    The `LaTeX` pane allows rendering LaTeX equations. It uses either
    `MathJax` or `KaTeX` depending on the defined renderer.

    By default it will use the renderer loaded in the extension
    (e.g. `pn.extension('katex')`), defaulting to `KaTeX`.

    Reference: https://panel.holoviz.org/reference/panes/LaTeX.html

    :Example:

    >>> pn.extension('katex')
    >>> LaTeX(
    ...     'The LaTeX pane supports two delimiters: $LaTeX$ and \(LaTeX\)',
    ...     styles={'font-size': '18pt'}, width=800
    ... )
    