from __future__ import annotations
import typing
from copy import copy
import pandas as pd
from matplotlib.animation import ArtistAnimation
from .exceptions import PlotnineError
def _draw_plots(self, plots: Iterable[ggplot]) -> tuple[Figure, list[list[Artist]]]:
    with pd.option_context('mode.chained_assignment', None):
        return self.__draw_plots(plots)