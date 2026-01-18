from typing import Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.timeline import core, types, drawings
from qiskit.visualization.timeline.plotters.base_plotter import BasePlotter
from qiskit.visualization.utils import matplotlib_close_if_inline
def _time_bucket_outline(self, xvals: np.ndarray, yvals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate outline of time bucket. Edges are smoothly faded.

        Args:
            xvals: Left and right point coordinates.
            yvals: Bottom and top point coordinates.

        Returns:
            Coordinate vectors of time bucket fringe.
        """
    x0, x1 = xvals
    y0, y1 = yvals
    width = x1 - x0
    y_mid = 0.5 * (y0 + y1)
    risefall = int(min(self.canvas.formatter['time_bucket.edge_dt'], max(width / 2 - 2, 0)))
    edge = np.sin(np.pi / 2 * np.arange(0, risefall) / risefall)
    xs = np.concatenate([np.arange(x0, x0 + risefall), [x0 + risefall, x1 - risefall], np.arange(x1 - risefall + 1, x1 + 1)])
    l1 = (y1 - y_mid) * np.concatenate([edge, [1, 1], edge[::-1]])
    l2 = (y0 - y_mid) * np.concatenate([edge, [1, 1], edge[::-1]])
    return (xs, l1, l2)