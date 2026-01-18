from __future__ import annotations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymatgen.util.plotting import pretty_plot
def add_electrode(self, electrode, label=None):
    """Add an electrode to the plot.

        Args:
            electrode: An electrode. All electrodes satisfying the
                AbstractElectrode interface should work.
            label: A label for the electrode. If None, defaults to a counting
                system, i.e. 'Electrode 1', 'Electrode 2', ...
        """
    if not label:
        label = f'Electrode {len(self._electrodes) + 1}'
    self._electrodes[label] = electrode