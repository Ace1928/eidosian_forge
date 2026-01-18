from pennylane.ops import Conditional
from pennylane.measurements import MidMeasureMP, MeasurementProcess
from .utils import default_wire_map
def _recursive_find_layer(layer_to_check, op_occupied_wires, occupied_wires_per_layer):
    """Determine correct layer for an operation drawn over ``op_occupied_wires``.

    An "occupied wire" will have something on top of it in the final drawing.  This could
    be a wire used by an operation or a wire between those used in a multi-qubit gate.

    In this function, we work with wires that are ordered, sequential integers, not the general
    hashable label that operations act on. ``drawable_layers`` performs this conversion.

    Args:
        layer_to_check (int): the function determines if the operation fits on this layer
        op_occupied_wires (set(int)): wires covered the drawn operation.  Includes everything
            between used wires in a multi-wire gate.
        occupied_wires_per_layer (list[set[int]]): which wires already have something drawn
            on them. Each set is a different layer.

    Returns:
        int: layer to place operation in
    """
    if occupied_wires_per_layer[layer_to_check] & op_occupied_wires:
        return layer_to_check + 1
    if layer_to_check == 0:
        return 0
    return _recursive_find_layer(layer_to_check - 1, op_occupied_wires, occupied_wires_per_layer)