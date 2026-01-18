import functools
import operator
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Union, TYPE_CHECKING
import numpy as np
import sympy
from ply import yacc
from cirq import ops, Circuit, NamedQubit, CX
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import.exception import QasmException
class QasmGateStatement:
    """Specifies how to convert a call to an OpenQASM gate
    to a list of `cirq.GateOperation`s.

    Has the responsibility to validate the arguments
    and parameters of the call and to generate a list of corresponding
    `cirq.GateOperation`s in the `on` method.
    """

    def __init__(self, qasm_gate: str, cirq_gate: Union[ops.Gate, Callable[[List[float]], ops.Gate]], num_params: int, num_args: int):
        """Initializes a Qasm gate statement.

        Args:
            qasm_gate: The symbol of the QASM gate.
            cirq_gate: The gate class on the cirq side.
            num_params: The number of params taken by this gate.
            num_args: The number of qubits (used in validation) this
                gate takes.
        """
        self.qasm_gate = qasm_gate
        self.cirq_gate = cirq_gate
        self.num_params = num_params
        assert num_args >= 1
        self.num_args = num_args

    def _validate_args(self, args: List[List[ops.Qid]], lineno: int):
        if len(args) != self.num_args:
            raise QasmException(f'{self.qasm_gate} only takes {self.num_args} arg(s) (qubits and/or registers), got: {len(args)}, at line {lineno}')

    def _validate_params(self, params: List[float], lineno: int):
        if len(params) != self.num_params:
            raise QasmException(f'{self.qasm_gate} takes {self.num_params} parameter(s), got: {len(params)}, at line {lineno}')

    def on(self, params: List[float], args: List[List[ops.Qid]], lineno: int) -> Iterable[ops.Operation]:
        self._validate_args(args, lineno)
        self._validate_params(params, lineno)
        reg_sizes = np.unique([len(reg) for reg in args])
        if len(reg_sizes) > 2 or (len(reg_sizes) > 1 and reg_sizes[0] != 1):
            raise QasmException(f'Non matching quantum registers of length {reg_sizes} at line {lineno}')
        final_gate: ops.Gate = self.cirq_gate if isinstance(self.cirq_gate, ops.Gate) else self.cirq_gate(params)
        op_qubits = functools.reduce(cast(Callable[[List['cirq.Qid'], List['cirq.Qid']], List['cirq.Qid']], np.broadcast), args)
        for qubits in op_qubits:
            if isinstance(qubits, ops.Qid):
                yield final_gate.on(qubits)
            elif len(np.unique(qubits)) < len(qubits):
                raise QasmException(f'Overlapping qubits in arguments at line {lineno}')
            else:
                yield final_gate.on(*qubits)