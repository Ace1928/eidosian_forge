from __future__ import annotations
from typing import TYPE_CHECKING
from .element_base import element_base

    theme element: line

    used for backgrounds and borders

    parameters
    ----------
    color : str | tuple
        line color
    colour : str | tuple
        alias of color
    linetype : str | tuple
        line style. if a string, it should be one of *solid*, *dashed*,
        *dashdot* or *dotted*. you can create interesting dashed patterns
        using tuples, see [](`~matplotlib.lines.line2D.set_linestyle`).
    size : float
        line thickness
    kwargs : dict
        Parameters recognised by [](`~matplotlib.lines.line2d`).
    