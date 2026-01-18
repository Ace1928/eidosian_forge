import copy
from collections.abc import Iterable
from typing import Optional, Union, Sequence
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import MeasurementShapeError, MeasurementTransform, Shadow, ShadowExpval
class ClassicalShadowMP(MeasurementTransform):
    """Represents a classical shadow measurement process occurring at the end of a
    quantum variational circuit.

    Please refer to :func:`classical_shadow` for detailed documentation.


    Args:
        wires (.Wires): The wires the measurement process applies to.
        seed (Union[int, None]): The seed used to generate the random measurements
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    def __init__(self, wires: Optional[Wires]=None, seed: Optional[int]=None, id: Optional[str]=None):
        self.seed = seed
        super().__init__(wires=wires, id=id)

    def _flatten(self):
        metadata = (('wires', self.wires), ('seed', self.seed))
        return ((None, None), metadata)

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (self.__class__.__name__, self.seed, tuple(self.wires.tolist()))
        return hash(fingerprint)

    def process(self, tape, device):
        """
        Returns the measured bits and recipes in the classical shadow protocol.

        The protocol is described in detail in the `classical shadows paper <https://arxiv.org/abs/2002.08953>`_.
        This measurement process returns the randomized Pauli measurements (the ``recipes``)
        that are performed for each qubit and snapshot as an integer:

        - 0 for Pauli X,
        - 1 for Pauli Y, and
        - 2 for Pauli Z.

        It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
        is sampled, and 1 if the -1 eigenvalue is sampled.

        The device shots are used to specify the number of snapshots. If ``T`` is the number
        of shots and ``n`` is the number of qubits, then both the measured bits and the
        Pauli measurements have shape ``(T, n)``.

        This implementation is device-agnostic and works by executing single-shot
        quantum tapes containing randomized Pauli observables. Devices should override this
        if they can offer cleaner or faster implementations.

        .. seealso:: :func:`~pennylane.classical_shadow`

        Args:
            tape (QuantumTape): the quantum tape to be processed
            device (pennylane.Device): the device used to process the quantum tape

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
        wires = self.wires
        n_snapshots = device.shots
        seed = self.seed
        with qml.workflow.set_shots(device, shots=1):
            n_qubits = len(wires)
            mapped_wires = np.array(device.map_wires(wires))
            rng = np.random.RandomState(seed)
            recipes = rng.randint(0, 3, size=(n_snapshots, n_qubits))
            obs_list = [qml.X, qml.Y, qml.Z]
            outcomes = np.zeros((n_snapshots, n_qubits))
            for t in range(n_snapshots):
                rotations = [rot for wire_idx, wire in enumerate(wires) for rot in obs_list[recipes[t][wire_idx]].compute_diagonalizing_gates(wires=wire)]
                device.reset()
                device.apply(tape.operations, rotations=tape.diagonalizing_gates + rotations)
                outcomes[t] = device.generate_samples()[0][mapped_wires]
        return qml.math.cast(qml.math.stack([outcomes, recipes]), dtype=np.int8)

    def process_state_with_shots(self, state: Sequence[complex], wire_order: Wires, shots: int, rng=None):
        """Process the given quantum state with the given number of shots

        Args:
            state (Sequence[complex]): quantum state vector given as a rank-N tensor, where
                each dim has size 2 and N is the number of wires.
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
            shots (int): The number of shots
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used. The random measurement outcomes
                in the form of bits will be generated from this argument, while the random recipes will be
                created from the ``seed`` argument provided to ``.ClassicalShadowsMP``.

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
        wire_map = {w: i for i, w in enumerate(wire_order)}
        mapped_wires = [wire_map[w] for w in self.wires]
        n_qubits = len(mapped_wires)
        num_dev_qubits = len(state.shape)
        recipe_rng = np.random.RandomState(self.seed)
        recipes = recipe_rng.randint(0, 3, size=(shots, n_qubits))
        bit_rng = np.random.default_rng(rng)
        obs_list = np.stack([qml.X.compute_matrix(), qml.Y.compute_matrix(), qml.Z.compute_matrix()])
        diag_list = np.stack([qml.Hadamard.compute_matrix(), qml.Hadamard.compute_matrix() @ qml.RZ.compute_matrix(-np.pi / 2), qml.Identity.compute_matrix()])
        obs = obs_list[recipes]
        diagonalizers = diag_list[recipes]
        unmeasured_wires = [i for i in range(num_dev_qubits) if i not in mapped_wires]
        transposed_state = np.transpose(state, axes=mapped_wires + unmeasured_wires)
        outcomes = np.zeros((shots, n_qubits))
        stacked_state = np.repeat(transposed_state[np.newaxis, ...], shots, axis=0)
        for active_qubit in range(n_qubits):
            num_remaining_qubits = num_dev_qubits - active_qubit
            conj_state_first_qubit = ABC[num_remaining_qubits]
            stacked_dim = ABC[num_remaining_qubits + 1]
            state_str = f'{stacked_dim}{ABC[:num_remaining_qubits]}'
            conj_state_str = f'{stacked_dim}{conj_state_first_qubit}{ABC[1:num_remaining_qubits]}'
            target_str = f'{stacked_dim}a{conj_state_first_qubit}'
            first_qubit_state = np.einsum(f'{state_str},{conj_state_str}->{target_str}', stacked_state, np.conj(stacked_state))
            probs = (np.einsum('abc,acb->a', first_qubit_state, obs[:, active_qubit]) + 1) / 2
            samples = bit_rng.random(size=probs.shape) > probs
            outcomes[:, active_qubit] = samples
            rotated_state = np.einsum('ab...,acb->ac...', stacked_state, diagonalizers[:, active_qubit])
            stacked_state = rotated_state[np.arange(shots), samples.astype(np.int8)]
            sum_indices = tuple(range(1, num_remaining_qubits))
            state_squared = np.abs(stacked_state) ** 2
            norms = np.sqrt(np.sum(state_squared, sum_indices, keepdims=True))
            stacked_state /= norms
        return np.stack([outcomes, recipes]).astype(np.int8)

    @property
    def samples_computational_basis(self):
        return False

    @property
    def numeric_type(self):
        return int

    @property
    def return_type(self):
        return Shadow

    def shape(self, device, shots):
        if not shots:
            raise MeasurementShapeError('Shots must be specified to obtain the shape of a classical shadow measurement process.')
        return (2, shots.total_shots, len(self.wires))

    def __copy__(self):
        return self.__class__(seed=self.seed, wires=self._wires)