import functools
import itertools
from collections import defaultdict
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
import pennylane.math as qnp
from pennylane import (
from pennylane.measurements import CountsMP, MutualInfoMP, SampleMP, StateMP, VnEntropyMP, PurityMP
from pennylane.operation import Channel
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.wires import Wires
from .._version import __version__
def _conjugate_state_with(k):
    """Perform the double tensor product k @ self._state @ k.conj().
            The `axes_left` and `axes_right` arguments are taken from the ambient variable space
            and `axes_right` is assumed to incorporate the tensor product and the transposition
            of k.conj() simultaneously."""
    return qnp.tensordot(qnp.tensordot(k, self._state, axes_left), qnp.conj(k), axes_right)