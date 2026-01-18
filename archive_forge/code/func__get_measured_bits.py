from collections import namedtuple
from functools import singledispatch
import pennylane as qml
from pennylane import ops
from pennylane.measurements import MidMeasureMP
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
from .style import _set_style
def _get_measured_bits(measurements, bit_map, offset):
    measured_bits = []
    for m in measurements:
        if isinstance(m.mv, list):
            for mv in m.mv:
                measured_bits += [bit_map[mcm] + offset for mcm in mv.measurements]
        elif m.mv:
            measured_bits += [bit_map[mcm] + offset for mcm in m.mv.measurements]
    return measured_bits