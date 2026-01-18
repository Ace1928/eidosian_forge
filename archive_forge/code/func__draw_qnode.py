from functools import wraps
from importlib.metadata import distribution
import warnings
import pennylane as qml
from .tape_mpl import tape_mpl
from .tape_text import tape_text
def _draw_qnode(qnode, wire_order=None, show_all_wires=False, decimals=2, max_length=100, show_matrices=True, expansion_strategy=None):

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        if isinstance(qnode.device, qml.devices.Device) and (expansion_strategy == 'device' or getattr(qnode, 'expansion_strategy', None) == 'device'):
            qnode.construct(args, kwargs)
            tapes = qnode.transform_program([qnode.tape])[0]
            program, _ = qnode.device.preprocess()
            tapes = program(tapes)[0]
        else:
            original_expansion_strategy = getattr(qnode, 'expansion_strategy', None)
            try:
                qnode.expansion_strategy = expansion_strategy or original_expansion_strategy
                tapes = qnode.construct(args, kwargs)
                program = qnode.transform_program
                tapes = program([qnode.tape])[0]
            finally:
                qnode.expansion_strategy = original_expansion_strategy
        _wire_order = wire_order or qnode.device.wires or qnode.tape.wires
        if tapes is not None:
            cache = {'tape_offset': 0, 'matrices': []}
            res = [tape_text(t, wire_order=_wire_order, show_all_wires=show_all_wires, decimals=decimals, show_matrices=False, max_length=max_length, cache=cache) for t in tapes]
            if show_matrices and cache['matrices']:
                mat_str = ''
                for i, mat in enumerate(cache['matrices']):
                    mat_str += f'\nM{i} = \n{mat}'
                if mat_str:
                    mat_str = '\n' + mat_str
                return '\n\n'.join(res) + mat_str
            return '\n\n'.join(res)
        return tape_text(qnode.qtape, wire_order=_wire_order, show_all_wires=show_all_wires, decimals=decimals, show_matrices=show_matrices, max_length=max_length)
    return wrapper