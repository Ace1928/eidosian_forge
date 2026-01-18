from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
class QuirkOp:
    """An operation as understood by Quirk's parser.

    Basically just a series of text identifiers for each qubit, and some rules
    for how things can be combined.
    """

    def __init__(self, *keys: Any, can_merge: bool=True) -> None:
        """Inits QuirkOp.

        Args:
            *keys: The JSON object(s) that each qubit is turned into when
                explaining a gate to Quirk. For example, a CNOT is turned into
                the keys ["•", "X"].

                Note that, when keys terminates early, it is implied that later
                qubits should use the same key as the last key.
            can_merge: Whether or not it is safe to merge a column containing
                this operation into a column containing other operations. For
                example, this is not safe if the column contains a control
                because the control would also apply to the other column's
                gates.
        """
        self.keys = keys
        self.can_merge = can_merge

    def controlled(self, control_count: int=1) -> 'QuirkOp':
        return QuirkOp(*['•'] * control_count, *self.keys, can_merge=False)