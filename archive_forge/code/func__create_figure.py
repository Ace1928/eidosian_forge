from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def _create_figure(self):
    """Create the main matplotlib figure."""
    self._figure = pyplot.figure(figsize=(self.ngates * self.scale, self.nqubits * self.scale), facecolor='w', edgecolor='w')
    ax = self._figure.add_subplot(1, 1, 1, frameon=True)
    ax.set_axis_off()
    offset = 0.5 * self.scale
    ax.set_xlim(self._gate_grid[0] - offset, self._gate_grid[-1] + offset)
    ax.set_ylim(self._wire_grid[0] - offset, self._wire_grid[-1] + offset)
    ax.set_aspect('equal')
    self._axes = ax