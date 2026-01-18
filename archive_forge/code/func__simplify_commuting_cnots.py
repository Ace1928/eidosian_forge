import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
def _simplify_commuting_cnots(cnots: List[Tuple[int, int]], flip_control_and_target: bool) -> Tuple[bool, List[Tuple[int, int]]]:
    """Attempts to commute CNOTs and remove cancelling pairs.

    Commutation relations are based on 9 (flip_control_and_target=False) or 10
    (flip_control_target=True) of [4]:
    When flip_control_target=True:

         CNOT(j, i) @ CNOT(j, k) = CNOT(j, k) @ CNOT(j, i)
    ───X───────       ───────X───
       │                     │
    ───@───@───   =   ───@───@───
           │             │
    ───────X───       ───X───────

    When flip_control_target=False:

    CNOT(i, j) @ CNOT(k, j) = CNOT(k, j) @ CNOT(i, j)
    ───@───────       ───────@───
       │                     │
    ───X───X───   =   ───X───X───
           │             │
    ───────@───       ───@───────

    Args:
        cnots: A list of CNOTS, encoded as integer tuples (control, target). The code does not make
            any assumption as to the order of the CNOTs, but it is likely to work better if its
            inputs are from Gray-sorted Hamiltonians. Regardless of the order of the CNOTs, the
            code is conservative and should be robust to mis-ordered inputs with the only side
            effect being a lack of simplification.
        flip_control_and_target: Whether to flip control and target.

    Returns:
        A tuple containing a Boolean that tells whether a simplification has been performed and the
        CNOT list, potentially simplified, encoded as integer tuples (control, target).
    """
    target, control = (0, 1) if flip_control_and_target else (1, 0)
    to_remove = set()
    qubit_to_index: List[Tuple[int, Dict[int, int]]] = []
    for j in range(len(cnots)):
        if not qubit_to_index or cnots[j][target] != qubit_to_index[-1][0]:
            qubit_to_index.append((cnots[j][target], {cnots[j][control]: j}))
            continue
        if cnots[j][control] in qubit_to_index[-1][1]:
            k = qubit_to_index[-1][1].pop(cnots[j][control])
            to_remove.update([k, j])
            if not qubit_to_index[-1][1]:
                qubit_to_index.pop()
        else:
            qubit_to_index[-1][1][cnots[j][control]] = j
    cnots = [cnot for i, cnot in enumerate(cnots) if i not in to_remove]
    return (bool(to_remove), cnots)