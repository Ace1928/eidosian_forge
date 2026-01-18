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
class TwoQubitInteractionHeatmap(Heatmap):
    """Visualizing interactions between neighboring qubits on a 2D grid."""

    def __init__(self, value_map: Mapping[QubitTuple, SupportsFloat], **kwargs):
        """Heatmap to display two-qubit interaction fidelities.

        Draw 2D qubit-qubit interaction heatmap with Matplotlib with arguments to configure the
        properties of the plot. The valid argument list includes all arguments of cirq.vis.Heatmap()
        plus the following.

        Args:
            value_map: A map from a qubit tuple location to a value.
            **kwargs: Optional kwargs including
                coupler_margin: float, default = 0.03
                coupler_width: float, default = 0.6
        """
        self._config: Dict[str, Any] = {'coupler_margin': 0.03, 'coupler_width': 0.6}
        super().__init__(value_map, **kwargs)

    def _extra_valid_kwargs(self) -> List[str]:
        return ['coupler_margin', 'coupler_width']

    def _qubits_to_polygon(self, qubits: QubitTuple) -> Tuple[Polygon, Point]:
        coupler_margin = self._config['coupler_margin']
        coupler_width = self._config['coupler_width']
        cwidth = coupler_width / 2.0
        setback = 0.5 - cwidth
        row1, col1 = map(float, (qubits[0].row, qubits[0].col))
        row2, col2 = map(float, (qubits[1].row, qubits[1].col))
        if abs(row1 - row2) + abs(col1 - col2) != 1:
            raise ValueError(f'{qubits[0]}-{qubits[1]} is not supported because they are not nearest neighbors')
        if coupler_width <= 0:
            polygon: Polygon = []
        elif row1 == row2:
            col1, col2 = (min(col1, col2), max(col1, col2))
            col_center = (col1 + col2) / 2.0
            polygon = [(col1 + coupler_margin, row1), (col_center - setback, row1 + cwidth - coupler_margin), (col_center + setback, row1 + cwidth - coupler_margin), (col2 - coupler_margin, row2), (col_center + setback, row1 - cwidth + coupler_margin), (col_center - setback, row1 - cwidth + coupler_margin)]
        elif col1 == col2:
            row1, row2 = (min(row1, row2), max(row1, row2))
            row_center = (row1 + row2) / 2.0
            polygon = [(col1, row1 + coupler_margin), (col1 + cwidth - coupler_margin, row_center - setback), (col1 + cwidth - coupler_margin, row_center + setback), (col2, row2 - coupler_margin), (col1 - cwidth + coupler_margin, row_center + setback), (col1 - cwidth + coupler_margin, row_center - setback)]
        return (polygon, Point((col1 + col2) / 2.0, (row1 + row2) / 2.0))

    def plot(self, ax: Optional[plt.Axes]=None, **kwargs: Any) -> Tuple[plt.Axes, mpl_collections.Collection]:
        """Plots the heatmap on the given Axes.
        Args:
            ax: the Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **kwargs: The optional keyword arguments are used to temporarily
                override the values present in the heatmap config. See
                __init__ for more details on the allowed arguments.
        Returns:
            A 2-tuple ``(ax, collection)``. ``ax`` is the `plt.Axes` that
            is plotted on. ``collection`` is the collection of paths drawn and filled.
        """
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax = cast(plt.Axes, ax)
        original_config = copy.deepcopy(self._config)
        self.update_config(**kwargs)
        qubits = set([q for qubits in self._value_map.keys() for q in qubits])
        Heatmap({q: 0.0 for q in qubits}).plot(ax=ax, collection_options={'cmap': 'binary', 'linewidths': 2, 'edgecolor': 'lightgrey', 'linestyle': 'dashed'}, plot_colorbar=False, annotation_format=None)
        collection = self._plot_on_axis(ax)
        if show_plot:
            fig.show()
        self._config = original_config
        return (ax, collection)