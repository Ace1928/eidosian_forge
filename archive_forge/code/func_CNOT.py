from collections.abc import Iterable
import warnings
from typing import Sequence
def CNOT(self, layer, wires, control_values=None, options=None):
    """Draws a CNOT gate.

        Args:
            layer (int): layer to draw in
            control_values=None (Union[bool, Iterable[bool]]): for each control wire, denotes whether to control
                on ``False=0`` or ``True=1``
            wires (Union[int, Iterable[int]]): wires to use. Last wire is the target.

        Keyword Args:
            options=None: Matplotlib options. The only supported keys are ``'color'``, ``'linewidth'``,
                and ``'zorder'``.

        **Example**

        .. code-block:: python

            drawer = MPLDrawer(n_wires=2, n_layers=2)

            drawer.CNOT(0, (0, 1))

            options = {'color': 'indigo', 'linewidth': 4}
            drawer.CNOT(1, (1, 0), options=options)

        .. figure:: ../../_static/drawer/cnot.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        """
    self.ctrl(layer, wires[:-1], wires[-1], control_values=control_values, options=options)
    self._target_x(layer, wires[-1], options=options)