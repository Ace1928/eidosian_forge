from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import re
import numpy as np
from cirq import ops, linalg, protocols, value
def _write_qasm(self, output_func: Callable[[str], None]) -> None:
    self.args.validate_version('2.0')
    line_gap = [0]

    def output_line_gap(n):
        line_gap[0] = max(line_gap[0], n)

    def output(text):
        if line_gap[0] > 0:
            output_func('\n' * line_gap[0])
            line_gap[0] = 0
        output_func(text)
    if self.header:
        for line in self.header.split('\n'):
            output(('// ' + line).rstrip() + '\n')
        output('\n')
    output('OPENQASM 2.0;\n')
    output('include "qelib1.inc";\n')
    output_line_gap(2)
    output(f'// Qubits: [{', '.join(map(str, self.qubits))}]\n')
    if len(self.qubits) > 0:
        output(f'qreg q[{len(self.qubits)}];\n')
    already_output_keys: Set[str] = set()
    for meas in self.measurements:
        key = protocols.measurement_key_name(meas)
        if key in already_output_keys:
            continue
        already_output_keys.add(key)
        meas_id = self.args.meas_key_id_map[key]
        comment = self.meas_comments[key]
        if comment is None:
            output(f'creg {meas_id}[{len(meas.qubits)}];\n')
        else:
            output(f'creg {meas_id}[{len(meas.qubits)}];  // Measurement: {comment}\n')
    output_line_gap(2)
    self._write_operations(self.operations, output, output_line_gap)