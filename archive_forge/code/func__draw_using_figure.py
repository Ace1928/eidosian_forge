from __future__ import annotations
import typing
from collections.abc import Sequence
from copy import copy, deepcopy
from io import BytesIO
from itertools import chain
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, Iterable, Optional
from warnings import warn
from ._utils import (
from ._utils.context import plot_context
from ._utils.ipython import (
from .coords import coord_cartesian
from .exceptions import PlotnineError, PlotnineWarning
from .facets import facet_null
from .facets.layout import Layout
from .geoms.geom_blank import geom_blank
from .guides.guides import guides
from .iapi import mpl_save_view
from .layer import Layers
from .mapping.aes import aes, make_labels
from .options import get_option
from .scales.scales import Scales
from .themes.theme import theme, theme_get
def _draw_using_figure(self, figure: Figure, axs: list[Axes]) -> ggplot:
    """
        Draw onto already created figure and axes

        This is can be used to draw animation frames,
        or inset plots. It is intended to be used
        after the key plot has been drawn.

        Parameters
        ----------
        figure :
            Matplotlib figure
        axs :
            Array of Axes onto which to draw the plots
        """
    from ._mpl.layout_engine import PlotnineLayoutEngine
    self = deepcopy(self)
    self.figure = figure
    self.axs = axs
    with plot_context(self):
        self._build()
        self.figure, self.axs = self.facet.setup(self)
        self.guides._setup(self)
        self.theme.setup(self)
        self._draw_layers()
        self._draw_breaks_and_labels()
        self.guides.draw()
        self.theme.apply()
        self.figure.set_layout_engine(PlotnineLayoutEngine(self))
    return self