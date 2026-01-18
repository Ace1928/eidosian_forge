from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
class NoiseModel(metaclass=value.ABCMetaImplementAnyOneOf):
    """Replaces operations and moments with noisy counterparts.

    A child class must override *at least one* of the following three methods:

        noisy_moments
        noisy_moment
        noisy_operation

    The methods that are not overridden will be implemented in terms of the ones
    that are.

    Simulators told to use a noise model will use these methods in order to
    dynamically rewrite the program they are simulating.
    """

    @classmethod
    def from_noise_model_like(cls, noise: 'cirq.NOISE_MODEL_LIKE') -> 'cirq.NoiseModel':
        """Transforms an object into a noise model if unambiguously possible.

        Args:
            noise: `None`, a `cirq.NoiseModel`, or a single qubit operation.

        Returns:
            `cirq.NO_NOISE` when given `None`,
            `cirq.ConstantQubitNoiseModel(gate)` when given a single qubit
            gate, or the given value if it is already a `cirq.NoiseModel`.

        Raises:
            ValueError: If noise is a `cirq.Gate` that acts on more than one
                qubit.
            TypeError: The input is not a ``cirq.NOISE_MODE_LIKE``.
        """
        if noise is None:
            return NO_NOISE
        if isinstance(noise, NoiseModel):
            return noise
        if isinstance(noise, ops.Gate):
            if noise.num_qubits() != 1:
                raise ValueError('Multi-qubit gates cannot be implicitly wrapped into a noise model. Please use a single qubit gate (which will be wrapped with `cirq.ConstantQubitNoiseModel`) or an instance of `cirq.NoiseModel`.')
            return ConstantQubitNoiseModel(noise)
        raise TypeError(f'Expected a NOISE_MODEL_LIKE (None, a cirq.NoiseModel, or a single qubit gate). Got {noise!r}')

    def is_virtual_moment(self, moment: 'cirq.Moment') -> bool:
        """Returns true iff the given moment is non-empty and all of its
        operations are virtual.

        Moments for which this method returns True should not have additional
        noise applied to them.

        Args:
            moment: ``cirq.Moment`` to check for non-virtual operations.

        Returns:
            True if "moment" is non-empty and all operations in "moment" are
            virtual; false otherwise.
        """
        if not moment.operations:
            return False
        return all((ops.VirtualTag() in op.tags for op in moment))

    def _noisy_moments_impl_moment(self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']) -> Sequence['cirq.OP_TREE']:
        result = []
        for moment in moments:
            result.append(self.noisy_moment(moment, system_qubits))
        return result

    def _noisy_moments_impl_operation(self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']) -> Sequence['cirq.OP_TREE']:
        result = []
        for moment in moments:
            result.append([self.noisy_operation(op) for op in moment])
        return result

    @value.alternative(requires='noisy_moment', implementation=_noisy_moments_impl_moment)
    @value.alternative(requires='noisy_operation', implementation=_noisy_moments_impl_operation)
    def noisy_moments(self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']) -> Sequence['cirq.OP_TREE']:
        """Adds possibly stateful noise to a series of moments.

        Args:
            moments: The moments to add noise to.
            system_qubits: A list of all qubits in the system.

        Returns:
            A sequence of OP_TREEs, with the k'th tree corresponding to the
            noisy operations for the k'th moment.
        """
        raise NotImplementedError

    def _noisy_moment_impl_moments(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return self.noisy_moments([moment], system_qubits)

    def _noisy_moment_impl_operation(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return [self.noisy_operation(op) for op in moment]

    @value.alternative(requires='noisy_moments', implementation=_noisy_moment_impl_moments)
    @value.alternative(requires='noisy_operation', implementation=_noisy_moment_impl_operation)
    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        """Adds noise to the operations from a moment.

        Args:
            moment: The moment to add noise to.
            system_qubits: A list of all qubits in the system.

        Returns:
            An OP_TREE corresponding to the noisy operations for the moment.
        """
        raise NotImplementedError

    def _noisy_operation_impl_moments(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        return self.noisy_moments([moment_module.Moment([operation])], operation.qubits)

    def _noisy_operation_impl_moment(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        return self.noisy_moment(moment_module.Moment([operation]), operation.qubits)

    @value.alternative(requires='noisy_moments', implementation=_noisy_operation_impl_moments)
    @value.alternative(requires='noisy_moment', implementation=_noisy_operation_impl_moment)
    def noisy_operation(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        """Adds noise to an individual operation.

        Args:
            operation: The operation to make noisy.

        Returns:
            An OP_TREE corresponding to the noisy operations implementing the
            noisy version of the given operation.
        """
        raise NotImplementedError