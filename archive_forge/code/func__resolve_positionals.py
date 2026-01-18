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
def _resolve_positionals(self, args: tuple[DataSource | VariableSpec, ...], data: DataSource, variables: dict[str, VariableSpec]) -> tuple[DataSource, dict[str, VariableSpec]]:
    """Handle positional arguments, which may contain data / x / y."""
    if len(args) > 3:
        err = 'Plot() accepts no more than 3 positional arguments (data, x, y).'
        raise TypeError(err)
    if isinstance(args[0], (abc.Mapping, pd.DataFrame)) or hasattr(args[0], '__dataframe__'):
        if data is not None:
            raise TypeError('`data` given by both name and position.')
        data, args = (args[0], args[1:])
    if len(args) == 2:
        x, y = args
    elif len(args) == 1:
        x, y = (*args, None)
    else:
        x = y = None
    for name, var in zip('yx', (y, x)):
        if var is not None:
            if name in variables:
                raise TypeError(f'`{name}` given by both name and position.')
            variables = {name: cast(VariableSpec, var), **variables}
    return (data, variables)