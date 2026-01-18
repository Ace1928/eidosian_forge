from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def _create_grid(self):
    """Create the grid of wires."""
    scale = self.scale
    wire_grid = np.arange(0.0, self.nqubits * scale, scale, dtype=float)
    gate_grid = np.arange(0.0, self.ngates * scale, scale, dtype=float)
    self._wire_grid = wire_grid
    self._gate_grid = gate_grid