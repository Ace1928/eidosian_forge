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
def _plot_layer(self, p: Plot, layer: Layer) -> None:
    data = layer['data']
    mark = layer['mark']
    move = layer['move']
    default_grouping_vars = ['col', 'row', 'group']
    grouping_properties = [v for v in PROPERTIES if v[0] not in 'xy']
    pair_variables = p._pair_spec.get('structure', {})
    for subplots, df, scales in self._generate_pairings(data, pair_variables):
        orient = layer['orient'] or mark._infer_orient(scales)

        def get_order(var):
            if var not in 'xy' and var in scales:
                return getattr(scales[var], 'order', None)
        if orient in df:
            width = pd.Series(index=df.index, dtype=float)
            for view in subplots:
                view_idx = self._get_subplot_data(df, orient, view, p._shares.get(orient)).index
                view_df = df.loc[view_idx]
                if 'width' in mark._mappable_props:
                    view_width = mark._resolve(view_df, 'width', None)
                elif 'width' in df:
                    view_width = view_df['width']
                else:
                    view_width = 0.8
                spacing = scales[orient]._spacing(view_df.loc[view_idx, orient])
                width.loc[view_idx] = view_width * spacing
            df['width'] = width
        if 'baseline' in mark._mappable_props:
            baseline = mark._resolve(df, 'baseline', None)
        else:
            baseline = 0 if 'baseline' not in df else df['baseline']
        df['baseline'] = baseline
        if move is not None:
            moves = move if isinstance(move, list) else [move]
            for move_step in moves:
                move_by = getattr(move_step, 'by', None)
                if move_by is None:
                    move_by = grouping_properties
                move_groupers = [*move_by, *default_grouping_vars]
                if move_step.group_by_orient:
                    move_groupers.insert(0, orient)
                order = {var: get_order(var) for var in move_groupers}
                groupby = GroupBy(order)
                df = move_step(df, groupby, orient, scales)
        df = self._unscale_coords(subplots, df, orient)
        grouping_vars = mark._grouping_props + default_grouping_vars
        split_generator = self._setup_split_generator(grouping_vars, df, subplots)
        mark._plot(split_generator, scales, orient)
    for view in self._subplots:
        view['ax'].autoscale_view()
    if layer['legend']:
        self._update_legend_contents(p, mark, data, scales, layer['label'])