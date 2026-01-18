import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr
from cirq.qis import quantum_state_representation
from cirq.value import big_endian_int_to_digits, linear_dict, random_state
class StabilizerState(quantum_state_representation.QuantumStateRepresentation, metaclass=abc.ABCMeta):
    """Interface for quantum stabilizer state representations.

    This interface is used for CliffordTableau and StabilizerChForm quantum
    state representations, allowing simulators to act on them abstractly.
    """

    @abc.abstractmethod
    def apply_x(self, axis: int, exponent: float=1, global_shift: float=0):
        """Apply an X operation to the state.

        Args:
            axis: The axis to which the operation should be applied.
            exponent: The exponent of the X operation, must be a half-integer.
            global_shift: The global phase shift of the raw operation, prior to
                exponentiation. Typically the value in `gate.global_shift`.
        Raises:
            ValueError: If the exponent is not half-integer.
        """

    @abc.abstractmethod
    def apply_y(self, axis: int, exponent: float=1, global_shift: float=0):
        """Apply an Y operation to the state.

        Args:
            axis: The axis to which the operation should be applied.
            exponent: The exponent of the Y operation, must be a half-integer.
            global_shift: The global phase shift of the raw operation, prior to
                exponentiation. Typically the value in `gate.global_shift`.
        Raises:
            ValueError: If the exponent is not half-integer.
        """

    @abc.abstractmethod
    def apply_z(self, axis: int, exponent: float=1, global_shift: float=0):
        """Apply a Z operation to the state.

        Args:
            axis: The axis to which the operation should be applied.
            exponent: The exponent of the Z operation, must be a half-integer.
            global_shift: The global phase shift of the raw operation, prior to
                exponentiation. Typically the value in `gate.global_shift`.
        Raises:
            ValueError: If the exponent is not half-integer.
        """

    @abc.abstractmethod
    def apply_h(self, axis: int, exponent: float=1, global_shift: float=0):
        """Apply an H operation to the state.

        Args:
            axis: The axis to which the operation should be applied.
            exponent: The exponent of the H operation, must be an integer.
            global_shift: The global phase shift of the raw operation, prior to
                exponentiation. Typically the value in `gate.global_shift`.
        Raises:
            ValueError: If the exponent is not an integer.
        """

    @abc.abstractmethod
    def apply_cz(self, control_axis: int, target_axis: int, exponent: float=1, global_shift: float=0):
        """Apply a CZ operation to the state.

        Args:
            control_axis: The control axis of the operation.
            target_axis: The axis to which the operation should be applied.
            exponent: The exponent of the CZ operation, must be an integer.
            global_shift: The global phase shift of the raw operation, prior to
                exponentiation. Typically the value in `gate.global_shift`.
        Raises:
            ValueError: If the exponent is not an integer.
        """

    @abc.abstractmethod
    def apply_cx(self, control_axis: int, target_axis: int, exponent: float=1, global_shift: float=0):
        """Apply a CX operation to the state.

        Args:
            control_axis: The control axis of the operation.
            target_axis: The axis to which the operation should be applied.
            exponent: The exponent of the CX operation, must be an integer.
            global_shift: The global phase shift of the raw operation, prior to
                exponentiation. Typically the value in `gate.global_shift`.
        Raises:
            ValueError: If the exponent is not an integer.
        """

    @abc.abstractmethod
    def apply_global_phase(self, coefficient: linear_dict.Scalar):
        """Apply a global phase to the state.

        Args:
            coefficient: The global phase to apply.
        """