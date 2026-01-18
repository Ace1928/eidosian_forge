from collections import namedtuple
from functools import singledispatch
import pennylane as qml
from pennylane import ops
from pennylane.measurements import MidMeasureMP
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
from .style import _set_style
def _tape_mpl(tape, wire_order=None, show_all_wires=False, decimals=None, *, fig=None, **kwargs):
    """Private function wrapped with styling."""
    wire_options = kwargs.get('wire_options', None)
    label_options = kwargs.get('label_options', None)
    active_wire_notches = kwargs.get('active_wire_notches', True)
    fontsize = kwargs.get('fontsize', None)
    wire_map = convert_wire_order(tape, wire_order=wire_order, show_all_wires=show_all_wires)
    tape = qml.map_wires(tape, wire_map=wire_map)[0][0]
    bit_map = default_bit_map(tape)
    layers = drawable_layers(tape.operations, wire_map={i: i for i in tape.wires}, bit_map=bit_map)
    for i, layer in enumerate(layers):
        if any((isinstance(o, qml.measurements.MidMeasureMP) and o.reset for o in layer)):
            layers.insert(i + 1, [])
    n_layers = len(layers)
    n_wires = len(wire_map)
    cwire_layers, cwire_wires = cwire_connections(layers + [tape.measurements], bit_map)
    drawer = MPLDrawer(n_layers=n_layers, n_wires=n_wires, c_wires=len(bit_map), wire_options=wire_options, fig=fig)
    config = _Config(decimals=decimals, active_wire_notches=active_wire_notches, bit_map=bit_map, terminal_layers=[cl[-1] for cl in cwire_layers])
    if n_wires == 0:
        return (drawer.fig, drawer.ax)
    if fontsize is not None:
        drawer.fontsize = fontsize
    drawer.label(list(wire_map), text_options=label_options)
    _add_classical_wires(drawer, cwire_layers, cwire_wires)
    for layer, layer_ops in enumerate(layers):
        for op in layer_ops:
            _add_operation_to_drawer(op, drawer, layer, config)
    for wire in _get_measured_wires(tape.measurements, list(range(n_wires))):
        drawer.measure(n_layers, wire)
    measured_bits = _get_measured_bits(tape.measurements, bit_map, drawer.n_wires)
    if measured_bits:
        drawer.measure(n_layers, measured_bits)
    return (drawer.fig, drawer.ax)