from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def _plot_measured_wires(self):
    ismeasured = self._measurements()
    xstop = self._gate_grid[-1]
    dy = 0.04
    for im in ismeasured:
        xdata = (self._gate_grid[ismeasured[im]], xstop + self.scale)
        ydata = (self._wire_grid[im] + dy, self._wire_grid[im] + dy)
        line = Line2D(xdata, ydata, color='k', lw=self.linewidth)
        self._axes.add_line(line)
    for i, g in enumerate(self._gates()):
        if isinstance(g, (CGate, CGateS)):
            wires = g.controls + g.targets
            for wire in wires:
                if wire in ismeasured and self._gate_grid[i] > self._gate_grid[ismeasured[wire]]:
                    ydata = (min(wires), max(wires))
                    xdata = (self._gate_grid[i] - dy, self._gate_grid[i] - dy)
                    line = Line2D(xdata, ydata, color='k', lw=self.linewidth)
                    self._axes.add_line(line)