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
def _apply_channel(self, kraus, wires):
    """Apply a quantum channel specified by a list of Kraus operators to subsystems of the
        quantum state. For a unitary gate, there is a single Kraus operator.

        Args:
            kraus (list[array]): Kraus operators
            wires (Wires): target wires
        """
    channel_wires = self.map_wires(wires)
    rho_dim = 2 * self.num_wires
    num_ch_wires = len(channel_wires)
    kraus_dagger = [qnp.conj(qnp.transpose(k)) for k in kraus]
    kraus = qnp.stack(kraus)
    kraus_dagger = qnp.stack(kraus_dagger)
    kraus_shape = [len(kraus)] + [2] * num_ch_wires * 2
    kraus = qnp.cast(qnp.reshape(kraus, kraus_shape), dtype=self.C_DTYPE)
    kraus_dagger = qnp.cast(qnp.reshape(kraus_dagger, kraus_shape), dtype=self.C_DTYPE)
    state_indices = ABC[:rho_dim]
    row_wires_list = channel_wires.tolist()
    row_indices = ''.join(ABC_ARRAY[row_wires_list].tolist())
    col_wires_list = [w + self.num_wires for w in row_wires_list]
    col_indices = ''.join(ABC_ARRAY[col_wires_list].tolist())
    new_row_indices = ABC[rho_dim:rho_dim + num_ch_wires]
    new_col_indices = ABC[rho_dim + num_ch_wires:rho_dim + 2 * num_ch_wires]
    kraus_index = ABC[rho_dim + 2 * num_ch_wires:rho_dim + 2 * num_ch_wires + 1]
    new_state_indices = functools.reduce(lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]), zip(col_indices + row_indices, new_col_indices + new_row_indices), state_indices)
    einsum_indices = f'{kraus_index}{new_row_indices}{row_indices}, {state_indices},{kraus_index}{col_indices}{new_col_indices}->{new_state_indices}'
    self._state = qnp.einsum(einsum_indices, kraus, self._state, kraus_dagger)