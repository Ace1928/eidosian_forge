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