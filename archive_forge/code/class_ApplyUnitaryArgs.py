import warnings
from typing import Any, cast, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg, qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
class ApplyUnitaryArgs:
    """Arguments for performing an efficient left-multiplication by a unitary.

    The receiving object is expected to mutate `target_tensor` so that it
    contains the state after multiplication, and then return `target_tensor`.
    Alternatively, if workspace is required, the receiving object can overwrite
    `available_buffer` with the results and return `available_buffer`. Or, if
    the receiving object is attempting to be simple instead of fast, it can
    create an entirely new array and return that.

    Attributes:
        target_tensor: The input tensor that needs to be left-multiplied by
            the unitary effect of the receiving object. The tensor will
            have the shape (2, 2, 2, ..., 2). It usually corresponds to
            a multi-qubit superposition, but it could also be a multi-qubit
            unitary transformation or some other concept.
        available_buffer: Pre-allocated workspace with the same shape and
            dtype as the target tensor.
        axes: Which axes the unitary effect is being applied to (e.g. the
            qubits that the gate is operating on).
        subspaces: Which subspace (in the computational basis) the unitary
            effect is being applied to, on each axis. By default it applies
            to subspace 0..d-1 on each axis, where d is the dimension of the
            unitary effect on that axis. Subspaces on each axis must be
            representable as a slice, so the dimensions specified here need to
            have a consistent step size.
    """

    def __init__(self, target_tensor: np.ndarray, available_buffer: np.ndarray, axes: Iterable[int], subspaces: Optional[Sequence[Tuple[int, ...]]]=None):
        """Inits ApplyUnitaryArgs.

        Args:
            target_tensor: The input tensor that needs to be left-multiplied by
                the unitary effect of the receiving object. The tensor will
                have the shape (2, 2, 2, ..., 2). It usually corresponds to
                a multi-qubit superposition, but it could also be a multi-qubit
                unitary transformation or some other concept.
            available_buffer: Pre-allocated workspace with the same shape and
                dtype as the target tensor.
            axes: Which axes the unitary effect is being applied to (e.g. the
                qubits that the gate is operating on).
            subspaces: Which subspace (in the computational basis) the unitary
                effect is being applied to, on each axis. By default it applies
                to subspace 0..d-1 on each axis, where d is the dimension of
                the unitary effect on that axis. Subspaces on each axis must be
                representable as a slice, so the dimensions specified here need
                to have a consistent step size.
        Raises:
            ValueError: If the subspace count does not equal the axis count, if
                any subspace has zero dimensions, or if any subspace has
                dimensions specified without a consistent step size.
        """
        self.target_tensor = target_tensor
        self.available_buffer = available_buffer
        self.axes = tuple(axes)
        if subspaces is not None:
            if len(self.axes) != len(subspaces):
                raise ValueError('Subspace count does not match axis count.')
            for subspace, axis in zip(subspaces, self.axes):
                if any((s >= target_tensor.shape[axis] for s in subspace)):
                    raise ValueError('Subspace specified does not exist in axis.')
        self.slices = None if subspaces is None else tuple(map(_to_slice, subspaces))

    @staticmethod
    def default(num_qubits: Optional[int]=None, *, qid_shape: Optional[Tuple[int, ...]]=None) -> 'ApplyUnitaryArgs':
        """A default instance starting in state |0⟩.

        Specify exactly one argument.

        Args:
            num_qubits: The number of qubits to make space for in the state.
            qid_shape: The shape of the state, specifying the dimension of each
                qid.

        Raises:
            TypeError: If exactly neither `num_qubits` or `qid_shape` is provided or
                both are provided.
        """
        if (num_qubits is None) == (qid_shape is None):
            raise TypeError('Specify exactly one of num_qubits or qid_shape.')
        if num_qubits is not None:
            qid_shape = (2,) * num_qubits
        qid_shape = cast(Tuple[int, ...], qid_shape)
        num_qubits = len(qid_shape)
        state = qis.one_hot(index=(0,) * num_qubits, shape=qid_shape, dtype=np.complex128)
        return ApplyUnitaryArgs(state, np.empty_like(state), range(num_qubits))

    @classmethod
    def for_unitary(cls, num_qubits: Optional[int]=None, *, qid_shape: Optional[Tuple[int, ...]]=None) -> 'ApplyUnitaryArgs':
        """A default instance corresponding to an identity matrix.

        Specify exactly one argument.

        Args:
            num_qubits: The number of qubits to make space for in the state.
            qid_shape: A tuple representing the number of quantum levels of each
                qubit the identity matrix applies to. `qid_shape` is (2, 2, 2) for
                a three-qubit identity operation tensor.

        Raises:
            TypeError: If exactly neither `num_qubits` or `qid_shape` is provided or
                both are provided.
        """
        if (num_qubits is None) == (qid_shape is None):
            raise TypeError('Specify exactly one of num_qubits or qid_shape.')
        if num_qubits is not None:
            qid_shape = (2,) * num_qubits
        qid_shape = cast(Tuple[int, ...], qid_shape)
        num_qubits = len(qid_shape)
        state = qis.eye_tensor(qid_shape, dtype=np.complex128)
        return ApplyUnitaryArgs(state, np.empty_like(state), range(num_qubits))

    def with_axes_transposed_to_start(self) -> 'ApplyUnitaryArgs':
        """Returns a transposed view of the same arguments.

        Returns:
            A view over the same target tensor and available workspace, but
            with the numpy arrays transposed such that the axes field is
            guaranteed to equal `range(len(result.axes))`. This allows one to
            say e.g. `result.target_tensor[0, 1, 0, ...]` instead of
            `result.target_tensor[result.subspace_index(0b010)]`.
        """
        axis_set = set(self.axes)
        other_axes = [axis for axis in range(len(self.target_tensor.shape)) if axis not in axis_set]
        perm = (*self.axes, *other_axes)
        target_tensor = self.target_tensor.transpose(*perm)
        available_buffer = self.available_buffer.transpose(*perm)
        return ApplyUnitaryArgs(target_tensor, available_buffer, range(len(self.axes)))

    def _for_operation_with_qid_shape(self, indices: Iterable[int], slices: Tuple[Union[int, slice], ...]) -> 'ApplyUnitaryArgs':
        """Creates a sliced and transposed view of `self` appropriate for an
        operation with shape `qid_shape` on qubits with the given indices.

        Example:
            sub_args = args._for_operation_with_qid_shape(indices, (2, 2, 2))
            # Slice where the first qubit is |1>.
            sub_args.target_tensor[..., 1, :, :]

        Args:
            indices: Integer indices into `self.axes` specifying which qubits
                the operation applies to.
            slices: The slices of the operation, the subdimension in each qubit
                the operation applies to.

        Returns: A new `ApplyUnitaryArgs` where `sub_args.target_tensor` and
            `sub_args.available_buffer` are sliced and transposed views of
            `self.target_tensor` and `self.available_buffer` respectively.
        """
        slices = tuple((size if isinstance(size, slice) else slice(0, size) for size in slices))
        sub_axes = [self.axes[i] for i in indices]
        axis_set = set(sub_axes)
        other_axes = [axis for axis in range(len(self.target_tensor.shape)) if axis not in axis_set]
        ordered_axes = (*other_axes, *sub_axes)
        target_tensor = self.target_tensor.transpose(*ordered_axes)[..., *slices]
        available_buffer = self.available_buffer.transpose(*ordered_axes)[..., *slices]
        new_axes = range(len(other_axes), len(ordered_axes))
        return ApplyUnitaryArgs(target_tensor, available_buffer, new_axes)

    def subspace_index(self, little_endian_bits_int: int=0, *, big_endian_bits_int: int=0) -> Tuple[Union[slice, int, 'ellipsis'], ...]:
        """An index for the subspace where the target axes equal a value.

        Args:
            little_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The least significant
                bit of the integer is the desired bit for the first axis, and
                so forth in increasing order. Can't be specified at the same
                time as `big_endian_bits_int`.
            big_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The most significant
                bit of the integer is the desired bit for the first axis, and
                so forth in decreasing order. Can't be specified at the same
                time as `little_endian_bits_int`.

        Returns:
            A value that can be used to index into `target_tensor` and
            `available_buffer`, and manipulate only the part of Hilbert space
            corresponding to a given bit assignment.

        Example:
            If `target_tensor` is a 4 qubit tensor and `axes` is `[1, 3]` and
            then this method will return the following when given
            `little_endian_bits=0b01`:

                `(slice(None), 0, slice(None), 1, Ellipsis)`

            Therefore the following two lines would be equivalent:

                args.target_tensor[args.subspace_index(0b01)] += 1

                args.target_tensor[:, 0, :, 1] += 1
        """
        return linalg.slice_for_qubits_equal_to(self.axes, little_endian_qureg_value=little_endian_bits_int, big_endian_qureg_value=big_endian_bits_int, qid_shape=self.target_tensor.shape)