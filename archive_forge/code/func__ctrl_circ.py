from collections.abc import Iterable
import warnings
from typing import Sequence
def _ctrl_circ(self, layer, wires, options=None):
    """Draw a solid circle that indicates control on one.

        Acceptable keys in options dictionary:
          * zorder
          * color
          * linewidth
        """
    if options is None:
        options = {}
    if 'color' not in options:
        options['color'] = plt.rcParams['lines.color']
    if 'zorder' not in options:
        options['zorder'] = 3
    circ_ctrl = plt.Circle((layer, wires), radius=self._ctrl_rad, **options)
    self._ax.add_patch(circ_ctrl)