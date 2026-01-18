import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
class QuilFormatter(string.Formatter):
    """A unique formatter to correctly output values to QUIL."""

    def __init__(self, qubit_id_map: Dict['cirq.Qid', str], measurement_id_map: Dict[str, str]) -> None:
        """Inits QuilFormatter.

        Args:
            qubit_id_map: A dictionary {qubit, quil_output_string} for
            the proper QUIL output for each qubit.
            measurement_id_map: A dictionary {measurement_key,
            quil_output_string} for the proper QUIL output for each
            measurement key.
        """
        self.qubit_id_map = {} if qubit_id_map is None else qubit_id_map
        self.measurement_id_map = {} if measurement_id_map is None else measurement_id_map

    def format_field(self, value: Any, spec: str) -> str:
        if isinstance(value, cirq.ops.Qid):
            value = self.qubit_id_map[value]
        if isinstance(value, str) and spec == 'meas':
            value = self.measurement_id_map[value]
            spec = ''
        return super().format_field(value, spec)