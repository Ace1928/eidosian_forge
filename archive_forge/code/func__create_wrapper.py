import functools
import inspect
import os
import warnings
import pennylane as qml
def _create_wrapper(self, obj, *targs, wire_order=None, **tkwargs):
    """Create a wrapper function that, when evaluated, transforms
        ``obj`` according to transform arguments ``*targs`` and ``**tkwargs``
        """
    if isinstance(obj, qml.operation.Operator):
        if wire_order is not None:
            tkwargs['wire_order'] = wire_order
        wrapper = self.fn(obj, *targs, **tkwargs)
    elif isinstance(obj, qml.tape.QuantumScript):
        tape, verified_wire_order = self._make_tape(obj, wire_order)
        if wire_order is not None:
            tkwargs['wire_order'] = verified_wire_order
        wrapper = self.tape_fn(tape, *targs, **tkwargs)
    elif callable(obj):

        def wrapper(*args, **kwargs):
            nonlocal wire_order
            tape, verified_wire_order = self._make_tape(obj, wire_order, *args, **kwargs)
            if wire_order is not None or ('wire_order' in self._sig and isinstance(obj, qml.QNode)):
                tkwargs['wire_order'] = verified_wire_order
            if isinstance(tape, qml.operation.Operator):
                return self.fn(tape, *targs, **tkwargs)
            if self.is_qfunc_transform:
                return self.tape_fn(obj, *kwargs, **tkwargs)(*args, **kwargs)
            return self.tape_fn(tape, *targs, **tkwargs)
    else:
        raise OperationTransformError('Input is not an Operator, tape, QNode, or quantum function')
    return wrapper