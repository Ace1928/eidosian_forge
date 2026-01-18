import pytest
import sympy
import cirq
from cirq.contrib.quirk.export_to_quirk import circuit_to_quirk_url
class MysteryGate(cirq.testing.SingleQubitGate):

    def _has_mixture_(self):
        return True