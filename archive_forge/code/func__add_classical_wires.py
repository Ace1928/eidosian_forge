from collections import namedtuple
from functools import singledispatch
import pennylane as qml
from pennylane import ops
from pennylane.measurements import MidMeasureMP
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
from .style import _set_style
def _add_classical_wires(drawer, layers, wires):
    for cwire, (cwire_layers, layer_wires) in enumerate(zip(layers, wires), start=drawer.n_wires):
        xs, ys = ([], [])
        len_diff = len(cwire_layers) - len(layer_wires)
        if len_diff > 0:
            layer_wires += [cwire] * len_diff
        for l, w in zip(cwire_layers, layer_wires):
            xs.extend([l, l, l])
            ys.extend([cwire, w, cwire])
        drawer.classical_wire(xs, ys)