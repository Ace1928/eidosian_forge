import functools
import inspect
import os
import warnings
import pennylane as qml
@staticmethod
def _make_tape(obj, wire_order, *args, **kwargs):
    """Given an input object, which may be:

        - an object such as a tape or a operation, or
        - a callable such as a QNode or a quantum function
          (alongside the callable arguments ``args`` and ``kwargs``),

        this function constructs and returns the tape/operation
        represented by the object.

        The ``wire_order`` argument determines whether a custom wire ordering
        should be used. If not provided, the wire ordering defaults to the
        objects wire ordering accessed via ``obj.wires``.

        Returns:
            tuple[.QuantumTape, Wires]: returns the tape and the verified wire order
        """
    if isinstance(obj, qml.QNode):
        obj.construct(args, kwargs)
        tape = obj.qtape
        wires = obj.device.wires
    elif isinstance(obj, qml.tape.QuantumScript):
        tape = obj
        wires = tape.wires
    elif inspect.isclass(obj) and issubclass(obj, qml.operation.Operator):
        with qml.QueuingManager.stop_recording():
            tape = obj(*args, **kwargs)
        wires = tape.wires
    elif callable(obj):
        tape = qml.tape.make_qscript(obj)(*args, **kwargs)
        wires = tape.wires
        if len(tape.operations) == 0 and len(tape.measurements) == 0:
            raise OperationTransformError('Quantum function contains no quantum operations')
    wire_order = wires if wire_order is None else qml.wires.Wires(wire_order)
    if not set(tape.wires).issubset(wire_order):
        raise OperationTransformError(f'Wires in circuit {tape.wires.tolist()} are inconsistent with those in wire_order {wire_order.tolist()}')
    return (tape, wire_order)