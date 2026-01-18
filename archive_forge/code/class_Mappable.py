from __future__ import annotations
from dataclasses import dataclass, fields, field
import textwrap
from typing import Any, Callable, Union
from collections.abc import Generator
import numpy as np
import pandas as pd
import matplotlib as mpl
from numpy import ndarray
from pandas import DataFrame
from matplotlib.artist import Artist
from seaborn._core.scales import Scale
from seaborn._core.properties import (
from seaborn._core.exceptions import PlotSpecError
class Mappable:

    def __init__(self, val: Any=None, depend: str | None=None, rc: str | None=None, auto: bool=False, grouping: bool=True):
        """
        Property that can be mapped from data or set directly, with flexible defaults.

        Parameters
        ----------
        val : Any
            Use this value as the default.
        depend : str
            Use the value of this feature as the default.
        rc : str
            Use the value of this rcParam as the default.
        auto : bool
            The default value will depend on other parameters at compile time.
        grouping : bool
            If True, use the mapped variable to define groups.

        """
        if depend is not None:
            assert depend in PROPERTIES
        if rc is not None:
            assert rc in mpl.rcParams
        self._val = val
        self._rc = rc
        self._depend = depend
        self._auto = auto
        self._grouping = grouping

    def __repr__(self):
        """Nice formatting for when object appears in Mark init signature."""
        if self._val is not None:
            s = f'<{repr(self._val)}>'
        elif self._depend is not None:
            s = f'<depend:{self._depend}>'
        elif self._rc is not None:
            s = f'<rc:{self._rc}>'
        elif self._auto:
            s = '<auto>'
        else:
            s = '<undefined>'
        return s

    @property
    def depend(self) -> Any:
        """Return the name of the feature to source a default value from."""
        return self._depend

    @property
    def grouping(self) -> bool:
        return self._grouping

    @property
    def default(self) -> Any:
        """Get the default value for this feature, or access the relevant rcParam."""
        if self._val is not None:
            return self._val
        elif self._rc is not None:
            return mpl.rcParams.get(self._rc)