import functools
import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice, QutritBasisState, DeviceError
from pennylane.wires import WireError
from pennylane.devices.default_qubit_legacy import _get_slice
from .._version import __version__
def _apply_tshift(self, state, axes, inverse=False):
    """Applies a ternary Shift gate by rolling 1 unit along the axis specified in ``axes``.

        Rolling by 1 unit along the axis means that the :math:`|0 \rangle` state with index ``0`` is
        shifted to the :math:`|1 \rangle` state with index ``1``. Likewise, since rolling beyond
        the last index loops back to the first, :math:`|2 \rangle` is transformed to
        :math:`|0 \rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            inverse (bool): whether to apply the inverse operation

        Returns:
            array[complex]: output state
        """
    shift = -1 if inverse else 1
    return self._roll(state, shift, axes[0])