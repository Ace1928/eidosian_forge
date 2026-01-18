from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
class CircuitPlot:
    """A class for managing a circuit plot."""
    scale = 1.0
    fontsize = 20.0
    linewidth = 1.0
    control_radius = 0.05
    not_radius = 0.15
    swap_delta = 0.05
    labels: list[str] = []
    inits: dict[str, str] = {}
    label_buffer = 0.5

    def __init__(self, c, nqubits, **kwargs):
        if not np or not matplotlib:
            raise ImportError('numpy or matplotlib not available.')
        self.circuit = c
        self.ngates = len(self.circuit.args)
        self.nqubits = nqubits
        self.update(kwargs)
        self._create_grid()
        self._create_figure()
        self._plot_wires()
        self._plot_gates()
        self._finish()

    def update(self, kwargs):
        """Load the kwargs into the instance dict."""
        self.__dict__.update(kwargs)

    def _create_grid(self):
        """Create the grid of wires."""
        scale = self.scale
        wire_grid = np.arange(0.0, self.nqubits * scale, scale, dtype=float)
        gate_grid = np.arange(0.0, self.ngates * scale, scale, dtype=float)
        self._wire_grid = wire_grid
        self._gate_grid = gate_grid

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

    def _plot_wires(self):
        """Plot the wires of the circuit diagram."""
        xstart = self._gate_grid[0]
        xstop = self._gate_grid[-1]
        xdata = (xstart - self.scale, xstop + self.scale)
        for i in range(self.nqubits):
            ydata = (self._wire_grid[i], self._wire_grid[i])
            line = Line2D(xdata, ydata, color='k', lw=self.linewidth)
            self._axes.add_line(line)
            if self.labels:
                init_label_buffer = 0
                if self.inits.get(self.labels[i]):
                    init_label_buffer = 0.25
                self._axes.text(xdata[0] - self.label_buffer - init_label_buffer, ydata[0], render_label(self.labels[i], self.inits), size=self.fontsize, color='k', ha='center', va='center')
        self._plot_measured_wires()

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

    def _plot_gates(self):
        """Iterate through the gates and plot each of them."""
        for i, gate in enumerate(self._gates()):
            gate.plot_gate(self, i)

    def _measurements(self):
        """Return a dict ``{i:j}`` where i is the index of the wire that has
        been measured, and j is the gate where the wire is measured.
        """
        ismeasured = {}
        for i, g in enumerate(self._gates()):
            if getattr(g, 'measurement', False):
                for target in g.targets:
                    if target in ismeasured:
                        if ismeasured[target] > i:
                            ismeasured[target] = i
                    else:
                        ismeasured[target] = i
        return ismeasured

    def _finish(self):
        for o in self._figure.findobj():
            o.set_clip_on(False)

    def one_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a single qubit gate."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        self._axes.text(x, y, t, color='k', ha='center', va='center', bbox={'ec': 'k', 'fc': 'w', 'fill': True, 'lw': self.linewidth}, size=self.fontsize)

    def two_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a two qubit gate. Does not work yet.
        """
        print(self._gate_grid)
        print(self._wire_grid)

    def control_line(self, gate_idx, min_wire, max_wire):
        """Draw a vertical control line."""
        xdata = (self._gate_grid[gate_idx], self._gate_grid[gate_idx])
        ydata = (self._wire_grid[min_wire], self._wire_grid[max_wire])
        line = Line2D(xdata, ydata, color='k', lw=self.linewidth)
        self._axes.add_line(line)

    def control_point(self, gate_idx, wire_idx):
        """Draw a control point."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.control_radius
        c = Circle((x, y), radius * self.scale, ec='k', fc='k', fill=True, lw=self.linewidth)
        self._axes.add_patch(c)

    def not_point(self, gate_idx, wire_idx):
        """Draw a NOT gates as the circle with plus in the middle."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.not_radius
        c = Circle((x, y), radius, ec='k', fc='w', fill=False, lw=self.linewidth)
        self._axes.add_patch(c)
        l = Line2D((x, x), (y - radius, y + radius), color='k', lw=self.linewidth)
        self._axes.add_line(l)

    def swap_point(self, gate_idx, wire_idx):
        """Draw a swap point as a cross."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        d = self.swap_delta
        l1 = Line2D((x - d, x + d), (y - d, y + d), color='k', lw=self.linewidth)
        l2 = Line2D((x - d, x + d), (y + d, y - d), color='k', lw=self.linewidth)
        self._axes.add_line(l1)
        self._axes.add_line(l2)