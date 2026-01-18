import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def _what_type_of_qubit_does_it_use(program: Program) -> Tuple[bool, bool, List[Union[Qubit, QubitPlaceholder]]]:
    """Helper function to peruse through a program's qubits.

    This function will also enforce the condition that a Program uses either all placeholders
    or all instantiated qubits to avoid accidentally mixing the two. This function will warn
    if your program doesn't use any qubits.

    :return: tuple of (whether the program uses placeholder qubits, whether the program uses
        real qubits, a list of qubits ordered by their first appearance in the program)
    """
    has_placeholders = False
    has_real_qubits = False
    qubits = {}
    for instr in program:
        if isinstance(instr, Gate):
            for q in instr.qubits:
                qubits[q] = 1
                if isinstance(q, QubitPlaceholder):
                    has_placeholders = True
                elif isinstance(q, Qubit):
                    has_real_qubits = True
                else:
                    raise ValueError('Unknown qubit type {}'.format(q))
        elif isinstance(instr, Measurement):
            qubits[instr.qubit] = 1
            if isinstance(instr.qubit, QubitPlaceholder):
                has_placeholders = True
            elif isinstance(instr.qubit, Qubit):
                has_real_qubits = True
            else:
                raise ValueError('Unknown qubit type {}'.format(instr.qubit))
        elif isinstance(instr, Pragma):
            for arg in instr.args:
                if isinstance(arg, QubitPlaceholder):
                    qubits[arg] = 1
                    has_placeholders = True
                elif isinstance(arg, Qubit):
                    qubits[arg] = 1
                    has_real_qubits = True
    if not (has_placeholders or has_real_qubits):
        warnings.warn("Your program doesn't use any qubits")
    if has_placeholders and has_real_qubits:
        raise ValueError('Your program mixes instantiated qubits with placeholders')
    return (has_placeholders, has_real_qubits, cast(List[Union[Qubit, QubitPlaceholder]], list(qubits.keys())))