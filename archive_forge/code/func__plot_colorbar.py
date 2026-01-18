import copy
from dataclasses import astuple, dataclass
from typing import (
import matplotlib as mpl
import matplotlib.collections as mpl_collections
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import axes_grid1
from cirq.devices import grid_qubit
from cirq.vis import vis_utils
def _plot_colorbar(self, mappable: mpl.cm.ScalarMappable, ax: plt.Axes) -> mpl.colorbar.Colorbar:
    """Plots the colorbar. Internal."""
    colorbar_ax = axes_grid1.make_axes_locatable(ax).append_axes(position=self._config['colorbar_position'], size=self._config['colorbar_size'], pad=self._config['colorbar_pad'])
    position = self._config['colorbar_position']
    orien = 'vertical' if position in ('left', 'right') else 'horizontal'
    colorbar = cast(plt.Figure, ax.figure).colorbar(mappable, colorbar_ax, ax, orientation=orien, **self._config.get('colorbar_options', {}))
    colorbar_ax.tick_params(axis='y', direction='out')
    return colorbar