from collections.abc import Iterable
import warnings
from typing import Sequence
def SWAP(self, layer, wires, options=None):
    """Draws a SWAP gate

        Args:
            layer (int): layer to draw on
            wires (Tuple[int, int]): two wires the SWAP acts on

        Keyword Args:
            options=None (dict): matplotlib keywords for ``Line2D`` objects

        **Example**

        The ``options`` keyword can accept any
        `Line2D compatible keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        in a dictionary.

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=2)

            drawer.SWAP(0, (0, 1))

            swap_options = {"linewidth": 2, "color": "indigo"}
            drawer.SWAP(1, (0, 1), options=swap_options)

        .. figure:: ../../_static/drawer/SWAP.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
    if options is None:
        options = {}
    line = plt.Line2D((layer, layer), wires, **options)
    self._ax.add_line(line)
    for wire in wires:
        self._swap_x(layer, wire, options)