from collections import defaultdict
import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane import QubitDevice
from pennylane.measurements import Shots
from pennylane.resource import Resources
from .._version import __version__
def _apply_z(self, state, axes, **kwargs):
    return [0.0]