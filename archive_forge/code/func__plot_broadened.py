import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
@staticmethod
def _plot_broadened(ax: 'matplotlib.axes.Axes', energies: Sequence[float], all_y: np.ndarray, all_labels: Sequence[str], mplargs: Union[Dict, None]):
    """Plot DOS data with labels to axes

        This is separated into another function so that subclasses can
        manipulate broadening, labels etc in their plot() method."""
    if mplargs is None:
        mplargs = {}
    all_lines = ax.plot(energies, all_y.T, **mplargs)
    for line, label in zip(all_lines, all_labels):
        line.set_label(label)
    ax.legend()
    ax.set_xlim(left=min(energies), right=max(energies))
    ax.set_ylim(bottom=0)