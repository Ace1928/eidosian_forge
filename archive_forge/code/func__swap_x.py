from collections.abc import Iterable
import warnings
from typing import Sequence
def _swap_x(self, layer, wire, options=None):
    """Draw an x such as used in drawing a swap gate

        Args:
            layer (int): layer to draw on
            wires (int): wire to draw on

        Keyword Args:
            options=None (dict): matplotlib keywords for ``Line2D`` objects
        """
    if options is None:
        options = {}
    if 'zorder' not in options:
        options['zorder'] = 2
    l1 = plt.Line2D((layer - self._swap_dx, layer + self._swap_dx), (wire - self._swap_dx, wire + self._swap_dx), **options)
    l2 = plt.Line2D((layer - self._swap_dx, layer + self._swap_dx), (wire + self._swap_dx, wire - self._swap_dx), **options)
    self._ax.add_line(l1)
    self._ax.add_line(l2)