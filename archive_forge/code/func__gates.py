from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def _gates(self):
    """Create a list of all gates in the circuit plot."""
    gates = []
    if isinstance(self.circuit, Mul):
        for g in reversed(self.circuit.args):
            if isinstance(g, Gate):
                gates.append(g)
    elif isinstance(self.circuit, Gate):
        gates.append(self.circuit)
    return gates