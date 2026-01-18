import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def _write_quil(self, output_func: Callable[[str], None]) -> None:
    """Calls `output_func` for successive lines of QUIL output.

        Args:
            output_func: A function that accepts a string of QUIL. This will likely
                write the QUIL to a file.

        Returns:
            None.
        """
    if self._decompose_operation is None:
        return super()._write_quil(output_func)
    output_func('# Created using Cirq.\n\n')
    if len(self.measurements) > 0:
        measurements_declared: Set[str] = set()
        for m in self.measurements:
            key = cirq.measurement_key_name(m)
            if key in measurements_declared:
                continue
            measurements_declared.add(key)
            output_func(f'DECLARE {self.measurement_id_map[key]} BIT[{len(m.qubits)}]\n')
        output_func('\n')
    for main_op in self.operations:
        decomposed = self._decompose_operation(main_op)
        for decomposed_op in decomposed:
            output_func(self._op_to_quil(decomposed_op))