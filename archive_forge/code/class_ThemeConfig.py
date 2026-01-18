from __future__ import annotations
import io
import os
import re
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Literal, Optional, cast
from xml.etree import ElementTree
from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import get_layout_engine, set_layout_engine
from seaborn.utils import _version_predates
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette
from typing import TYPE_CHECKING, TypedDict
class ThemeConfig(mpl.RcParams):
    """
    Configuration object for the Plot.theme, using matplotlib rc parameters.
    """
    THEME_GROUPS = ['axes', 'figure', 'font', 'grid', 'hatch', 'legend', 'lines', 'mathtext', 'markers', 'patch', 'savefig', 'scatter', 'xaxis', 'xtick', 'yaxis', 'ytick']

    def __init__(self):
        super().__init__()
        self.reset()

    @property
    def _default(self) -> dict[str, Any]:
        return {**self._filter_params(mpl.rcParamsDefault), **axes_style('darkgrid'), **plotting_context('notebook'), 'axes.prop_cycle': cycler('color', color_palette('deep'))}

    def reset(self) -> None:
        """Update the theme dictionary with seaborn's default values."""
        self.update(self._default)

    def update(self, other: dict[str, Any] | None=None, /, **kwds):
        """Update the theme with a dictionary or keyword arguments of rc parameters."""
        if other is not None:
            theme = self._filter_params(other)
        else:
            theme = {}
        theme.update(kwds)
        super().update(theme)

    def _filter_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Restruct to thematic rc params."""
        return {k: v for k, v in params.items() if any((k.startswith(p) for p in self.THEME_GROUPS))}

    def _html_table(self, params: dict[str, Any]) -> list[str]:
        lines = ['<table>']
        for k, v in params.items():
            row = f"<tr><td>{k}:</td><td style='text-align:left'>{v!r}</td></tr>"
            lines.append(row)
        lines.append('</table>')
        return lines

    def _repr_html_(self) -> str:
        repr = ["<div style='height: 300px'>", "<div style='border-style: inset; border-width: 2px'>", *self._html_table(self), '</div>', '</div>']
        return '\n'.join(repr)