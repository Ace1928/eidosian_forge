from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class ParameterizableTag:

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def _is_parameterized_(self) -> bool:
        return cirq.is_parameterized(self.value)

    def _parameter_names_(self) -> AbstractSet[str]:
        return cirq.parameter_names(self.value)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'ParameterizableTag':
        return ParameterizableTag(cirq.resolve_parameters(self.value, resolver, recursive))