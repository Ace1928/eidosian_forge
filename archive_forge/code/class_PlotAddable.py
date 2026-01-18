from __future__ import annotations
import sys
from typing import (
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypeAlias  # noqa: TCH002
from plotnine import ggplot, guide_colorbar, guide_legend
from plotnine.iapi import strip_label_details
class PlotAddable(Protocol):
    """
    Object that can be added to a ggplot object
    """

    def __radd__(self, plot: ggplot) -> ggplot:
        """
        Add to ggplot object

        Parameters
        ----------
        other :
            ggplot object

        Returns
        -------
        :
            ggplot object
        """
        ...