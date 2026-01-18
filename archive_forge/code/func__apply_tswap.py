import functools
import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice, QutritBasisState, DeviceError
from pennylane.wires import WireError
from pennylane.devices.default_qubit_legacy import _get_slice
from .._version import __version__
def _apply_tswap(self, state, axes, **kwargs):
    """Applies a ternary SWAP gate by performing a partial transposition along the
        specified axes. The ternary SWAP gate acts on the computational basis states like
        :math:`\x0bert i, j\rangle \rightarrow \x0bert j, i \rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
    all_axes = list(range(len(state.shape)))
    all_axes[axes[0]] = axes[1]
    all_axes[axes[1]] = axes[0]
    return self._transpose(state, all_axes)