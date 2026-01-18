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
def _update_legend_contents(self, p: Plot, mark: Mark, data: PlotData, scales: dict[str, Scale], layer_label: str | None) -> None:
    """Add legend artists / labels for one layer in the plot."""
    if data.frame.empty and data.frames:
        legend_vars: list[str] = []
        for frame in data.frames.values():
            frame_vars = frame.columns.intersection(list(scales))
            legend_vars.extend((v for v in frame_vars if v not in legend_vars))
    else:
        legend_vars = list(data.frame.columns.intersection(list(scales)))
    if layer_label is not None:
        legend_title = str(p._labels.get('legend', ''))
        layer_key = (legend_title, -1)
        artist = mark._legend_artist([], None, {})
        if artist is not None:
            for content in self._legend_contents:
                if content[0] == layer_key:
                    content[1].append(artist)
                    content[2].append(layer_label)
                    break
            else:
                self._legend_contents.append((layer_key, [artist], [layer_label]))
    schema: list[tuple[tuple[str, str | int], list[str], tuple[list[Any], list[str]]]] = []
    schema = []
    for var in legend_vars:
        var_legend = scales[var]._legend
        if var_legend is not None:
            values, labels = var_legend
            for (_, part_id), part_vars, _ in schema:
                if data.ids[var] == part_id:
                    part_vars.append(var)
                    break
            else:
                title = self._resolve_label(p, var, data.names[var])
                entry = ((title, data.ids[var]), [var], (values, labels))
                schema.append(entry)
    contents: list[tuple[tuple[str, str | int], Any, list[str]]] = []
    for key, variables, (values, labels) in schema:
        artists = []
        for val in values:
            artist = mark._legend_artist(variables, val, scales)
            if artist is not None:
                artists.append(artist)
        if artists:
            contents.append((key, artists, labels))
    self._legend_contents.extend(contents)