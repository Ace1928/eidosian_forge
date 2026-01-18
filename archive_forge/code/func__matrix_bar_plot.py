import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _matrix_bar_plot(mat: np.ndarray, z_label: str, ax: mplot3d.axes3d.Axes3D, kets: Optional[Sequence[str]]=None, title: Optional[str]=None, ylim: Tuple[int, int]=(-1, 1), **bar3d_kwargs: Any) -> None:
    num_rows, num_cols = mat.shape
    indices = np.meshgrid(range(num_cols), range(num_rows))
    x_indices = np.array(indices[1]).flatten()
    y_indices = np.array(indices[0]).flatten()
    z_indices = np.zeros(mat.size)
    dx = np.ones(mat.size) * 0.3
    dy = np.ones(mat.size) * 0.3
    dz = mat.flatten()
    ax.bar3d(x_indices, y_indices, z_indices, dx, dy, dz, color='#ff0080', alpha=1.0, **bar3d_kwargs)
    ax.set_zlabel(z_label)
    ax.set_zlim3d(ylim[0], ylim[1])
    if kets is not None:
        ax.set_xticks(np.arange(num_cols) + 0.15)
        ax.set_yticks(np.arange(num_rows) + 0.15)
        ax.set_xticklabels(kets)
        ax.set_yticklabels(kets)
    if title is not None:
        ax.set_title(title)