from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def _add_grouping_symbols(op, layer_str, config):
    """Adds symbols indicating the extent of a given object."""
    if len(op.wires) > 1:
        mapped_wires = [config.wire_map[w] for w in op.wires]
        min_w, max_w = (min(mapped_wires), max(mapped_wires))
        layer_str[min_w] = '╭'
        layer_str[max_w] = '╰'
        for w in range(min_w + 1, max_w):
            layer_str[w] = '├' if w in mapped_wires else '│'
    return layer_str