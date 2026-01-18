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
def _get_polygon_units(self) -> List[PolygonUnit]:
    polygon_unit_list: List[PolygonUnit] = []
    for qubits, value in sorted(self._value_map.items()):
        polygon, center = self._qubits_to_polygon(qubits)
        polygon_unit_list.append(PolygonUnit(polygon=polygon, center=center, value=float(value), annot=self._get_annotation_value(qubits, value)))
    return polygon_unit_list