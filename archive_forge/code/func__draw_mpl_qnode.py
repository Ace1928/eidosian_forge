from functools import wraps
from importlib.metadata import distribution
import warnings
import pennylane as qml
from .tape_mpl import tape_mpl
from .tape_text import tape_text
def _draw_mpl_qnode(qnode, wire_order=None, show_all_wires=False, decimals=None, expansion_strategy=None, style='black_white', *, fig=None, **kwargs):

    @wraps(qnode)
    def wrapper(*args, **kwargs_qnode):
        if expansion_strategy == 'device' and isinstance(qnode.device, qml.devices.Device):
            qnode.construct(args, kwargs)
            tapes, _ = qnode.transform_program([qnode.tape])
            program, _ = qnode.device.preprocess()
            tapes, _ = program(tapes)
            tape = tapes[0]
        else:
            original_expansion_strategy = getattr(qnode, 'expansion_strategy', None)
            try:
                qnode.expansion_strategy = expansion_strategy or original_expansion_strategy
                qnode.construct(args, kwargs_qnode)
                program = qnode.transform_program
                [tape], _ = program([qnode.tape])
            finally:
                qnode.expansion_strategy = original_expansion_strategy
        _wire_order = wire_order or qnode.device.wires or tape.wires
        return tape_mpl(tape, wire_order=_wire_order, show_all_wires=show_all_wires, decimals=decimals, style=style, fig=fig, **kwargs)
    return wrapper