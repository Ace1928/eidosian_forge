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
def _unscale_coords(self, subplots: list[dict], df: DataFrame, orient: str) -> DataFrame:
    coord_cols = [c for c in df if re.match('^[xy]\\D*$', str(c))]
    out_df = df.drop(coord_cols, axis=1).reindex(df.columns, axis=1).copy(deep=False)
    for view in subplots:
        view_df = self._filter_subplot_data(df, view)
        axes_df = view_df[coord_cols]
        for var, values in axes_df.items():
            axis = getattr(view['ax'], f'{str(var)[0]}axis')
            transform = axis.get_transform().inverted().transform
            inverted = transform(values)
            out_df.loc[values.index, str(var)] = inverted
    return out_df