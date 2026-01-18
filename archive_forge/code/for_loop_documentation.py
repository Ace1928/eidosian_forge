from __future__ import annotations
import warnings
from typing import Iterable, Optional, Union, TYPE_CHECKING
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from .control_flow import ControlFlowOp
A context manager for building up ``for`` loops onto circuits in a natural order, without
    having to construct the loop body first.

    Within the block, a lot of the bookkeeping is done for you; you do not need to keep track of
    which qubits and clbits you are using, for example, and a loop parameter will be allocated for
    you, if you do not supply one yourself.  All normal methods of accessing the qubits on the
    underlying :obj:`~QuantumCircuit` will work correctly, and resolve into correct accesses within
    the interior block.

    You generally should never need to instantiate this object directly.  Instead, use
    :obj:`.QuantumCircuit.for_loop` in its context-manager form, i.e. by not supplying a ``body`` or
    sets of qubits and clbits.

    Example usage::

        import math
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)) as i:
            qc.rx(i * math.pi/4, 0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, True)

    This context should almost invariably be created by a :meth:`.QuantumCircuit.for_loop` call, and
    the resulting instance is a "friend" of the calling circuit.  The context will manipulate the
    circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and exited
    (by popping its scope, building it, and appending the resulting :obj:`.ForLoopOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    