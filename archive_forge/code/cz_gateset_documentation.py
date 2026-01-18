from typing import Any, Dict, Sequence, Type, Union, TYPE_CHECKING
from cirq import ops, protocols
from cirq.transformers.analytical_decompositions import two_qubit_to_cz
from cirq.transformers.target_gatesets import compilation_target_gateset
Initializes CZTargetGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            allow_partial_czs: If set, all powers of the form `cirq.CZ**t`, and not just
             `cirq.CZ`, are part of this gateset.
            additional_gates: Sequence of additional gates / gate families which should also
              be "accepted" by this gateset. This is empty by default.
        