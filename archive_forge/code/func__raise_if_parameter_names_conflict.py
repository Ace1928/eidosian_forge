from __future__ import annotations
from typing import Callable, Union
import numbers
import operator
import numpy
import symengine
from qiskit.circuit.exceptions import CircuitError
def _raise_if_parameter_names_conflict(self, inbound_parameters, outbound_parameters=None):
    if outbound_parameters is None:
        outbound_parameters = set()
        outbound_names = {}
    else:
        outbound_names = {p.name: p for p in outbound_parameters}
    inbound_names = inbound_parameters
    conflicting_names = []
    for name, param in inbound_names.items():
        if name in self._names and name not in outbound_names:
            if param != self._names[name]:
                conflicting_names.append(name)
    if conflicting_names:
        raise CircuitError(f'Name conflict applying operation for parameters: {conflicting_names}')