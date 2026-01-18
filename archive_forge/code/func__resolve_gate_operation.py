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
def _resolve_gate_operation(self, args: List[List[ops.Qid]], gate: str, p: Any, params: List[float]):
    gate_set = self.basic_gates if not self.qelibinc else self.all_gates
    if gate not in gate_set.keys():
        tip = ', did you forget to include qelib1.inc?' if not self.qelibinc else ''
        msg = f'Unknown gate "{gate}" at line {p.lineno(1)}{tip}'
        raise QasmException(msg)
    p[0] = gate_set[gate].on(args=args, params=params, lineno=p.lineno(1))