from collections import namedtuple
from functools import singledispatch
import pennylane as qml
from pennylane import ops
from pennylane.measurements import MidMeasureMP
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
from .style import _set_style
def _get_measured_wires(measurements, wires) -> set:
    measured_wires = set()
    for m in measurements:
        if not m.mv:
            if len(m.wires) == 0:
                return wires
            for wire in m.wires:
                measured_wires.add(wire)
    return measured_wires