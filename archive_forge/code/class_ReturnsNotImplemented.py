import pytest, sympy
import cirq
from cirq.study import ParamResolver
class ReturnsNotImplemented:

    def _is_parameterized_(self):
        return NotImplemented

    def _resolve_parameters_(self, resolver, recursive):
        return NotImplemented