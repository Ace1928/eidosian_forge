import abc
import itertools
import warnings
from collections import defaultdict
from typing import Union, List
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.math import multiply as qmlmul
from pennylane.math import sum as qmlsum
from pennylane.measurements import (
from pennylane.resource import Resources
from pennylane.operation import operation_derivative, Operation
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
class QubitDevice(Device):
    """Abstract base class for PennyLane qubit devices.

    The following abstract method **must** be defined:

    * :meth:`~.apply`: append circuit operations, compile the circuit (if applicable),
      and perform the quantum computation.

    Devices that generate their own samples (such as hardware) may optionally
    overwrite :meth:`~.probability`. This method otherwise automatically
    computes the probabilities from the generated samples, and **must**
    overwrite the following method:

    * :meth:`~.generate_samples`: Generate samples from the device from the
      exact or approximate probability distribution.

    Analytic devices **must** overwrite the following method:

    * :meth:`~.analytic_probability`: returns the probability or marginal probability from the
      device after circuit execution. :meth:`~.marginal_prob` may be used here.

    This device contains common utility methods for qubit-based devices. These
    do not need to be overwritten. Utility methods include:

    * :meth:`~.expval`, :meth:`~.var`, :meth:`~.sample`: return expectation values,
      variances, and samples of observables after the circuit has been rotated
      into the observable eigenbasis.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        r_dtype: Real floating point precision type.
        c_dtype: Complex floating point precision type.
    """
    _asarray = staticmethod(np.asarray)
    _dot = staticmethod(np.dot)
    _abs = staticmethod(np.abs)
    _reduce_sum = staticmethod(lambda array, axes: np.sum(array, axis=tuple(axes)))
    _reshape = staticmethod(np.reshape)
    _flatten = staticmethod(lambda array: array.flatten())
    _gather = staticmethod(lambda array, indices, axis=0: array[:, indices] if axis == 1 else array[indices])
    _einsum = staticmethod(np.einsum)
    _cast = staticmethod(np.asarray)
    _transpose = staticmethod(np.transpose)
    _tensordot = staticmethod(np.tensordot)
    _conj = staticmethod(np.conj)
    _imag = staticmethod(np.imag)
    _roll = staticmethod(np.roll)
    _stack = staticmethod(np.stack)
    _outer = staticmethod(np.outer)
    _diag = staticmethod(np.diag)
    _real = staticmethod(np.real)
    _size = staticmethod(np.size)
    _ndim = staticmethod(np.ndim)

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        new_array = np.zeros(new_dimensions, dtype=array.dtype.type)
        new_array[indices] = array
        return new_array

    @staticmethod
    def _const_mul(constant, array):
        """Data type preserving multiply operation"""
        return qmlmul(constant, array, dtype=array.dtype)
    observables = {'PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'Hermitian', 'Identity', 'Projector', 'Sum', 'Sprod', 'Prod'}
    measurement_map = defaultdict(lambda: '')
    'Mapping used to override the logic of measurement processes. The dictionary maps a\n    measurement class to a string containing the name of a device\'s method that overrides the\n    measurement process. The method defined by the device should have the following arguments:\n\n    * measurement (MeasurementProcess): measurement to override\n    * shot_range (tuple[int]): 2-tuple of integers specifying the range of samples\n        to use. If not specified, all samples are used.\n    * bin_size (int): Divides the shot range into bins of size ``bin_size``, and\n        returns the measurement statistic separately over each bin. If not\n        provided, the entire shot range is treated as a single bin.\n\n    .. note::\n\n        When overriding the logic of a :class:`~pennylane.measurements.MeasurementTransform`, the\n        method defined by the device should only have a single argument:\n\n        * tape: quantum tape to transform\n\n    **Example:**\n\n    Let\'s create a device that inherits from :class:`~pennylane.devices.DefaultQubitLegacy` and overrides the\n    logic of the `qml.sample` measurement. To do so we will need to update the ``measurement_map``\n    dictionary:\n\n    .. code-block:: python\n\n        class NewDevice(DefaultQubitLegacy):\n            def __init__(self, wires, shots):\n                super().__init__(wires=wires, shots=shots)\n                self.measurement_map[SampleMP] = "sample_measurement"\n\n            def sample_measurement(self, measurement, shot_range=None, bin_size=None):\n                return 2\n\n    >>> dev = NewDevice(wires=2, shots=1000)\n    >>> @qml.qnode(dev)\n    ... def circuit():\n    ...     return qml.sample()\n    >>> circuit()\n    tensor(2, requires_grad=True)\n    '

    def __init__(self, wires=1, shots=None, *, r_dtype=np.float64, c_dtype=np.complex128, analytic=None):
        super().__init__(wires=wires, shots=shots, analytic=analytic)
        if 'float' not in str(r_dtype):
            raise DeviceError('Real datatype must be a floating point type.')
        if 'complex' not in str(c_dtype):
            raise DeviceError('Complex datatype must be a complex floating point type.')
        self.C_DTYPE = c_dtype
        self.R_DTYPE = r_dtype
        self._samples = None
        'None or array[int]: stores the samples generated by the device\n        *after* rotation to diagonalize the observables.'

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(model='qubit', supports_broadcasting=False, supports_finite_shots=True, supports_tensor_observables=True, returns_probs=True)
        return capabilities

    def reset(self):
        """Reset the backend state.

        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        self._samples = None

    def execute(self, circuit, **kwargs):
        """It executes a queue of quantum operations on the device and then measure the given observables.

        For plugin developers: instead of overwriting this, consider
        implementing a suitable subset of

        * :meth:`apply`

        * :meth:`~.generate_samples`

        * :meth:`~.probability`

        Additional keyword arguments may be passed to this method
        that can be utilised by :meth:`apply`. An example would be passing
        the ``QNode`` hash that can be used later for parametric compilation.

        Args:
            circuit (~.tape.QuantumTape): circuit to execute on the device

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            array[float]: measured value(s)
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Entry with args=(circuit=%s, kwargs=%s) called by=%s', circuit, kwargs, '::L'.join((str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3])))
        self.check_validity(circuit.operations, circuit.observables)
        self.apply(circuit.operations, rotations=self._get_diagonalizing_gates(circuit), **kwargs)
        sample_type = (SampleMP, CountsMP, ClassicalShadowMP, ShadowExpvalMP)
        if self.shots is not None or any((isinstance(m, sample_type) for m in circuit.measurements)):
            is_lightning = hasattr(self, 'name') and isinstance(self.name, str) and ('Lightning' in self.name) and getattr(self, '_CPP_BINARY_AVAILABLE', False)
            diagonalizing_gates = self._get_diagonalizing_gates(circuit) if is_lightning else None
            if is_lightning and diagonalizing_gates:
                self.apply(diagonalizing_gates)
            self._samples = self.generate_samples()
            if is_lightning and diagonalizing_gates:
                self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
        if self._shot_vector is not None:
            results = self.shot_vec_statistics(circuit)
        else:
            results = self.statistics(circuit)
            single_measurement = len(circuit.measurements) == 1
            results = results[0] if single_measurement else tuple(results)
        self._num_executions += 1
        if self.tracker.active:
            shots_from_dev = self._shots if not self.shot_vector else self._raw_shot_sequence
            tape_resources = circuit.specs['resources']
            resources = Resources(tape_resources.num_wires, tape_resources.num_gates, tape_resources.gate_types, tape_resources.gate_sizes, tape_resources.depth, Shots(shots_from_dev))
            self.tracker.update(executions=1, shots=self._shots, results=results, resources=resources)
            self.tracker.record()
        return results

    def shot_vec_statistics(self, circuit: QuantumTape):
        """Process measurement results from circuit execution using a device
        with a shot vector and return statistics.

        This is an auxiliary method of execute and uses statistics.

        When using shot vectors, measurement results for each item of the shot
        vector are contained in a tuple.

        Args:
            circuit (~.tape.QuantumTape): circuit to execute on the device

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            tuple: statistics for each shot item from the shot vector
        """
        results = []
        s1 = 0
        measurements = circuit.measurements
        counts_exist = any((isinstance(m, CountsMP) for m in measurements))
        single_measurement = len(measurements) == 1
        for shot_tuple in self._shot_vector:
            s2 = s1 + np.prod(shot_tuple)
            r = self.statistics(circuit, shot_range=[s1, s2], bin_size=shot_tuple.shots)
            if single_measurement:
                r = r[0]
            elif shot_tuple.copies == 1:
                r = tuple((r_[0] if isinstance(r_, list) else r_.T for r_ in r))
            elif counts_exist:
                r = self._multi_meas_with_counts_shot_vec(circuit, shot_tuple, r)
            else:
                r = [tuple((self._asarray(r_.T[idx]) for r_ in r)) for idx in range(shot_tuple.copies)]
            if isinstance(r, qml.numpy.ndarray):
                if shot_tuple.copies > 1:
                    results.extend([self._asarray(r_) for r_ in qml.math.unstack(r.T)])
                else:
                    results.append(r.T)
            elif single_measurement and counts_exist:
                results.extend(r)
            elif not single_measurement and shot_tuple.copies > 1:
                r = [tuple((elem if isinstance(elem, dict) else elem.T for elem in r_)) for r_ in r]
                results.extend(r)
            else:
                results.append(r)
            s1 = s2
        return tuple(results)

    def _multi_meas_with_counts_shot_vec(self, circuit: QuantumTape, shot_tuple, r):
        """Auxiliary function of the shot_vec_statistics and execute
        functions for post-processing the results of multiple measurements at
        least one of which was a counts measurement.

        The measurements were executed on a device that defines a shot vector.
        """
        new_r = []
        for idx in range(shot_tuple.copies):
            result_group = []
            for idx2, r_ in enumerate(r):
                measurement_proc = circuit.measurements[idx2]
                if isinstance(measurement_proc, ProbabilityMP) or (isinstance(measurement_proc, SampleMP) and measurement_proc.obs):
                    result = r_[:, idx]
                else:
                    result = r_[idx]
                if not isinstance(measurement_proc, CountsMP):
                    result = self._asarray(result.T)
                result_group.append(result)
            new_r.append(tuple(result_group))
        return new_r

    def batch_execute(self, circuits):
        """Execute a batch of quantum circuits on the device.

        The circuits are represented by tapes, and they are executed one-by-one using the
        device's ``execute`` method. The results are collected in a list.

        For plugin developers: This function should be overwritten if the device can efficiently run multiple
        circuits on a backend, for example using parallel and/or asynchronous executions.

        Args:
            circuits (list[~.tape.QuantumTape]): circuits to execute on the device

        Returns:
            list[array[float]]: list of measured value(s)
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Entry with args=(circuits=%s) called by=%s', circuits, '::L'.join((str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3])))
        results = []
        for circuit in circuits:
            self.reset()
            res = self.execute(circuit)
            results.append(res)
        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()
        return results

    @abc.abstractmethod
    def apply(self, operations, **kwargs):
        """Apply quantum operations, rotate the circuit into the measurement
        basis, and compile and execute the quantum circuit.

        This method receives a list of quantum operations queued by the QNode,
        and should be responsible for:

        * Constructing the quantum program
        * (Optional) Rotating the quantum circuit using the rotation
          operations provided. This diagonalizes the circuit so that arbitrary
          observables can be measured in the computational basis.
        * Compile the circuit
        * Execute the quantum circuit

        Both arguments are provided as lists of PennyLane :class:`~.Operation`
        instances. Useful properties include :attr:`~.Operation.name`,
        :attr:`~.Operation.wires`, and :attr:`~.Operation.parameters`:

        >>> op = qml.RX(0.2, wires=[0])
        >>> op.name # returns the operation name
        "RX"
        >>> op.wires # returns a Wires object representing the wires that the operation acts on
        <Wires = [0]>
        >>> op.parameters # returns a list of parameters
        [0.2]

        Args:
            operations (list[~.Operation]): operations to apply to the device

        Keyword args:
            rotations (list[~.Operation]): operations that rotate the circuit
                pre-measurement into the eigenbasis of the observables.
            hash (int): the hash value of the circuit constructed by `CircuitGraph.hash`
        """

    @staticmethod
    def active_wires(operators):
        """Returns the wires acted on by a set of operators.

        Args:
            operators (list[~.Operation]): operators for which
                we are gathering the active wires

        Returns:
            Wires: wires activated by the specified operators
        """
        list_of_wires = [op.wires for op in operators]
        return Wires.all_wires(list_of_wires)

    def _measure(self, measurement: Union[SampleMeasurement, StateMeasurement], shot_range=None, bin_size=None):
        """Compute the corresponding measurement process depending on ``shots`` and the measurement
        type.

        Args:
            measurement (Union[SampleMeasurement, StateMeasurement]): measurement process
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Raises:
            ValueError: if the measurement cannot be computed

        Returns:
            Union[float, dict, list[float]]: result of the measurement
        """
        if self.shots is None:
            if isinstance(measurement, StateMeasurement):
                return measurement.process_state(state=self.state, wire_order=self.wires)
            raise ValueError(f'Shots must be specified in the device to compute the measurement {measurement.__class__.__name__}')
        if isinstance(measurement, StateMeasurement):
            warnings.warn(f'Requested measurement {measurement.__class__.__name__} with finite shots; the returned state information is analytic and is unaffected by sampling. To silence this warning, set shots=None on the device.', UserWarning)
            return measurement.process_state(state=self.state, wire_order=self.wires)
        return measurement.process_samples(samples=self._samples, wire_order=self.wires, shot_range=shot_range, bin_size=bin_size)

    def statistics(self, circuit: QuantumTape, shot_range=None, bin_size=None):
        """Process measurement results from circuit execution and return statistics.

        This includes returning expectation values, variance, samples, probabilities, states, and
        density matrices.

        Args:
            circuit (~.tape.QuantumTape): the quantum tape currently being executed
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            Union[float, List[float]]: the corresponding statistics

        .. details::
            :title: Usage Details

            The ``shot_range`` and ``bin_size`` arguments allow for the statistics
            to be performed on only a subset of device samples. This finer level
            of control is accessible from the main UI by instantiating a device
            with a batch of shots.

            For example, consider the following device:

            >>> dev = qml.device("my_device", shots=[5, (10, 3), 100])

            This device will execute QNodes using 135 shots, however
            measurement statistics will be **course grained** across these 135
            shots:

            * All measurement statistics will first be computed using the
              first 5 shots --- that is, ``shots_range=[0, 5]``, ``bin_size=5``.

            * Next, the tuple ``(10, 3)`` indicates 10 shots, repeated 3 times. We will want to use
              ``shot_range=[5, 35]``, performing the expectation value in bins of size 10
              (``bin_size=10``).

            * Finally, we repeat the measurement statistics for the final 100 shots,
              ``shot_range=[35, 135]``, ``bin_size=100``.
        """
        measurements = circuit.measurements
        results = []
        for m in measurements:
            if isinstance(m.mv, list):
                obs = m
            else:
                obs = m.obs or m.mv or m
            if (method := getattr(self, self.measurement_map[type(m)], False)):
                if isinstance(m, MeasurementTransform):
                    result = method(tape=circuit)
                else:
                    result = method(m, shot_range=shot_range, bin_size=bin_size)
            elif isinstance(m, ExpectationMP):
                result = self.expval(obs, shot_range=shot_range, bin_size=bin_size)
            elif isinstance(m, VarianceMP):
                result = self.var(obs, shot_range=shot_range, bin_size=bin_size)
            elif isinstance(m, SampleMP):
                samples = self.sample(obs, shot_range=shot_range, bin_size=bin_size, counts=False)
                result = self._asarray(qml.math.squeeze(samples))
            elif isinstance(m, CountsMP):
                result = self.sample(m, shot_range=shot_range, bin_size=bin_size, counts=True)
            elif isinstance(m, ProbabilityMP):
                result = self.probability(wires=m.wires, shot_range=shot_range, bin_size=bin_size)
            elif isinstance(m, StateMP):
                if len(measurements) > 1:
                    raise qml.QuantumFunctionError('The state or density matrix cannot be returned in combination with other return types')
                if self.shots is not None:
                    warnings.warn('Requested state or density matrix with finite shots; the returned state information is analytic and is unaffected by sampling. To silence this warning, set shots=None on the device.', UserWarning)
                state = self.access_state(wires=obs.wires)
                result = self._asarray(state, dtype=self.C_DTYPE)
            elif isinstance(m, VnEntropyMP):
                if self.wires.labels != tuple(range(self.num_wires)):
                    raise qml.QuantumFunctionError('Returning the Von Neumann entropy is not supported when using custom wire labels')
                if self._shot_vector is not None:
                    raise NotImplementedError('Returning the Von Neumann entropy is not supported with shot vectors.')
                if self.shots is not None:
                    warnings.warn('Requested Von Neumann entropy with finite shots; the returned result is analytic and is unaffected by sampling. To silence this warning, set shots=None on the device.', UserWarning)
                result = self.vn_entropy(wires=obs.wires, log_base=obs.log_base)
            elif isinstance(m, MutualInfoMP):
                if self.wires.labels != tuple(range(self.num_wires)):
                    raise qml.QuantumFunctionError('Returning the mutual information is not supported when using custom wire labels')
                if self._shot_vector is not None:
                    raise NotImplementedError('Returning the mutual information is not supported with shot vectors.')
                if self.shots is not None:
                    warnings.warn('Requested mutual information with finite shots; the returned state information is analytic and is unaffected by sampling. To silence this warning, set shots=None on the device.', UserWarning)
                wires0, wires1 = obs.raw_wires
                result = self.mutual_info(wires0=wires0, wires1=wires1, log_base=obs.log_base)
            elif isinstance(m, ClassicalShadowMP):
                if len(measurements) > 1:
                    raise qml.QuantumFunctionError('Classical shadows cannot be returned in combination with other return types')
                result = self.classical_shadow(obs, circuit)
            elif isinstance(m, ShadowExpvalMP):
                if len(measurements) > 1:
                    raise qml.QuantumFunctionError('Classical shadows cannot be returned in combination with other return types')
                result = self.shadow_expval(obs, circuit=circuit)
            elif isinstance(m, MeasurementTransform):
                result = m.process(tape=circuit, device=self)
            elif isinstance(m, (SampleMeasurement, StateMeasurement)):
                result = self._measure(m, shot_range=shot_range, bin_size=bin_size)
            elif m.return_type is not None:
                name = obs.name if isinstance(obs, qml.operation.Operator) else type(obs).__name__
                raise qml.QuantumFunctionError(f'Unsupported return type specified for observable {name}')
            else:
                result = None
            if isinstance(m, (ExpectationMP, VarianceMP, ProbabilityMP, VnEntropyMP, MutualInfoMP, ShadowExpvalMP)):
                result = self._asarray(result, dtype=self.R_DTYPE)
            if self._shot_vector is not None and isinstance(result, np.ndarray):
                result = qml.math.squeeze(result)
            if result is not None:
                results.append(result)
        return results

    def access_state(self, wires=None):
        """Check that the device has access to an internal state and return it if available.

        Args:
            wires (Wires): wires of the reduced system

        Raises:
            QuantumFunctionError: if the device is not capable of returning the state

        Returns:
            array or tensor: the state or the density matrix of the device
        """
        if not self.capabilities().get('returns_state'):
            raise qml.QuantumFunctionError('The current device is not capable of returning the state')
        state = getattr(self, 'state', None)
        if state is None:
            raise qml.QuantumFunctionError('The state is not available in the current device')
        if wires:
            density_matrix = self.density_matrix(wires)
            return density_matrix
        return state

    def generate_samples(self):
        """Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\\dots,q_{N-1}\\rangle` where
        :math:`q_0` is the most significant bit.

        .. warning::

            This method should be overwritten on devices that
            generate their own computational basis samples, with the resulting
            computational basis samples stored as ``self._samples``.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        number_of_states = 2 ** self.num_wires
        rotated_prob = self.analytic_probability()
        samples = self.sample_basis_states(number_of_states, rotated_prob)
        return self.states_to_binary(samples, self.num_wires)

    def sample_basis_states(self, number_of_states, state_probability):
        """Sample from the computational basis states based on the state
        probability.

        This is an auxiliary method to the generate_samples method.

        Args:
            number_of_states (int): the number of basis states to sample from
            state_probability (array[float]): the computational basis probability vector

        Returns:
            array[int]: the sampled basis states
        """
        if self.shots is None:
            raise qml.QuantumFunctionError('The number of shots has to be explicitly set on the device when using sample-based measurements.')
        shots = self.shots
        basis_states = np.arange(number_of_states)
        if self._ndim(state_probability) == 2:
            return np.array([np.random.choice(basis_states, shots, p=prob) for prob in state_probability])
        return np.random.choice(basis_states, shots, p=state_probability)

    @staticmethod
    def generate_basis_states(num_wires, dtype=np.uint32):
        """
        Generates basis states in binary representation according to the number
        of wires specified.

        The states_to_binary method creates basis states faster (for larger
        systems at times over x25 times faster) than the approach using
        ``itertools.product``, at the expense of using slightly more memory.

        Due to the large size of the integer arrays for more than 32 bits,
        memory allocation errors may arise in the states_to_binary method.
        Hence we constraint the dtype of the array to represent unsigned
        integers on 32 bits. Due to this constraint, an overflow occurs for 32
        or more wires, therefore this approach is used only for fewer wires.

        For smaller number of wires speed is comparable to the next approach
        (using ``itertools.product``), hence we resort to that one for testing
        purposes.

        Args:
            num_wires (int): the number wires
            dtype=np.uint32 (type): the data type of the arrays to use

        Returns:
            array[int]: the sampled basis states
        """
        if 2 < num_wires < 32:
            states_base_ten = np.arange(2 ** num_wires, dtype=dtype)
            return QubitDevice.states_to_binary(states_base_ten, num_wires, dtype=dtype)
        basis_states_generator = itertools.product((0, 1), repeat=num_wires)
        return np.fromiter(itertools.chain(*basis_states_generator), dtype=int).reshape(-1, num_wires)

    @staticmethod
    def states_to_binary(samples, num_wires, dtype=np.int64):
        """Convert basis states from base 10 to binary representation.

        This is an auxiliary method to the generate_samples method.

        Args:
            samples (array[int]): samples of basis states in base 10 representation
            num_wires (int): the number of qubits
            dtype (type): Type of the internal integer array to be used. Can be
                important to specify for large systems for memory allocation
                purposes.

        Returns:
            array[int]: basis states in binary representation
        """
        powers_of_two = 1 << np.arange(num_wires, dtype=dtype)
        states_sampled_base_ten = samples[..., None] & powers_of_two
        return (states_sampled_base_ten > 0).astype(dtype)[..., ::-1]

    @property
    def circuit_hash(self):
        """The hash of the circuit upon the last execution.

        This can be used by devices in :meth:`~.apply` for parametric compilation.
        """
        raise NotImplementedError

    @property
    def state(self):
        """Returns the state vector of the circuit prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        raise NotImplementedError

    def density_matrix(self, wires):
        """Returns the reduced density matrix over the given wires.

        Args:
            wires (Wires): wires of the reduced system

        Returns:
            array[complex]: complex array of shape ``(2 ** len(wires), 2 ** len(wires))``
            representing the reduced density matrix of the state prior to measurement.
        """
        state = getattr(self, 'state', None)
        wires = self.map_wires(wires)
        return qml.math.reduce_statevector(state, indices=wires, c_dtype=self.C_DTYPE)

    def vn_entropy(self, wires, log_base):
        """Returns the Von Neumann entropy prior to measurement.

        .. math::
            S( \\rho ) = -\\text{Tr}( \\rho \\log ( \\rho ))

        Args:
            wires (Wires): Wires of the considered subsystem.
            log_base (float): Base for the logarithm, default is None the natural logarithm is used in this case.

        Returns:
            float: returns the Von Neumann entropy
        """
        try:
            state = self.density_matrix(wires=self.wires)
        except qml.QuantumFunctionError as e:
            raise NotImplementedError(f'Cannot compute the Von Neumman entropy with device {self.name} that is not capable of returning the state. ') from e
        wires = wires.tolist()
        return qml.math.vn_entropy(state, indices=wires, c_dtype=self.C_DTYPE, base=log_base)

    def mutual_info(self, wires0, wires1, log_base):
        """Returns the mutual information prior to measurement:

        .. math::

            I(A, B) = S(\\rho^A) + S(\\rho^B) - S(\\rho^{AB})

        where :math:`S` is the von Neumann entropy.

        Args:
            wires0 (Wires): wires of the first subsystem
            wires1 (Wires): wires of the second subsystem
            log_base (float): base to use in the logarithm

        Returns:
            float: the mutual information
        """
        try:
            state = self.density_matrix(wires=self.wires)
        except qml.QuantumFunctionError as e:
            raise NotImplementedError(f'Cannot compute the mutual information with device {self.name} that is not capable of returning the state. ') from e
        wires0 = wires0.tolist()
        wires1 = wires1.tolist()
        return qml.math.mutual_info(state, indices0=wires0, indices1=wires1, c_dtype=self.C_DTYPE, base=log_base)

    def classical_shadow(self, obs, circuit):
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
        tapes containing randomized Pauli observables. Devices should override this
        if they can offer cleaner or faster implementations.

        .. seealso:: :func:`~.pennylane.classical_shadow`

        Args:
            obs (~.pennylane.measurements.ClassicalShadowMP): The classical shadow measurement process
            circuit (~.tape.QuantumTape): The quantum tape that is being executed

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
        if circuit is None:
            raise ValueError('Circuit must be provided when measuring classical shadows')
        wires = obs.wires
        n_snapshots = self.shots
        seed = obs.seed
        with qml.workflow.set_shots(self, shots=1):
            n_qubits = len(wires)
            mapped_wires = np.array(self.map_wires(wires))
            rng = np.random.RandomState(seed)
            recipes = rng.randint(0, 3, size=(n_snapshots, n_qubits))
            obs_list = [qml.X, qml.Y, qml.Z]
            outcomes = np.zeros((n_snapshots, n_qubits))
            for t in range(n_snapshots):
                rotations = [rot for wire_idx, wire in enumerate(wires) for rot in obs_list[recipes[t][wire_idx]].compute_diagonalizing_gates(wires=wire)]
                self.reset()
                self.apply(circuit.operations, rotations=self._get_diagonalizing_gates(circuit) + rotations)
                outcomes[t] = self.generate_samples()[0][mapped_wires]
        return self._cast(self._stack([outcomes, recipes]), dtype=np.int8)

    def shadow_expval(self, obs, circuit):
        """Compute expectation values using classical shadows in a differentiable manner.

        Please refer to :func:`~.pennylane.shadow_expval` for detailed documentation.

        Args:
            obs (~.pennylane.measurements.ClassicalShadowMP): The classical shadow expectation
                value measurement process
            circuit (~.tape.QuantumTape): The quantum tape that is being executed

        Returns:
            float: expectation value estimate.
        """
        bits, recipes = self.classical_shadow(obs, circuit)
        shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=obs.wires.tolist())
        return shadow.expval(obs.H, obs.k)

    def analytic_probability(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        PennyLane uses the convention
        :math:`|q_0,q_1,\\dots,q_{N-1}\\rangle` where :math:`q_0` is the most
        significant bit.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.


        .. note::

            :meth:`~.marginal_prob` may be used as a utility method
            to calculate the marginal probability distribution.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """
        raise NotImplementedError

    def estimate_probability(self, wires=None, shot_range=None, bin_size=None):
        """Return the estimated probability of each computational basis state
        using the generated samples.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to calculate
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """
        wires = wires or self.wires
        wires = Wires(wires)
        device_wires = self.map_wires(wires)
        num_wires = len(device_wires)
        if shot_range is None:
            samples = self._samples[..., device_wires]
        else:
            samples = self._samples[..., slice(*shot_range), device_wires]
        powers_of_two = 2 ** np.arange(num_wires)[::-1]
        indices = samples @ powers_of_two
        batch_size = self._samples.shape[0] if np.ndim(self._samples) == 3 else None
        dim = 2 ** num_wires
        if bin_size is not None:
            num_bins = samples.shape[-2] // bin_size
            prob = self._count_binned_samples(indices, batch_size, dim, bin_size, num_bins)
        else:
            prob = self._count_unbinned_samples(indices, batch_size, dim)
        return self._asarray(prob, dtype=self.R_DTYPE)

    @staticmethod
    def _count_unbinned_samples(indices, batch_size, dim):
        """Count the occurences of sampled indices and convert them to relative
        counts in order to estimate their occurence probability."""
        if batch_size is None:
            prob = np.zeros(dim, dtype=np.float64)
            basis_states, counts = np.unique(indices, return_counts=True)
            prob[basis_states] = counts / len(indices)
            return prob
        prob = np.zeros((batch_size, dim), dtype=np.float64)
        for i, idx in enumerate(indices):
            basis_states, counts = np.unique(idx, return_counts=True)
            prob[i, basis_states] = counts / len(idx)
        return prob

    @staticmethod
    def _count_binned_samples(indices, batch_size, dim, bin_size, num_bins):
        """Count the occurences of bins of sampled indices and convert them to relative
        counts in order to estimate their occurence probability per bin."""
        if batch_size is None:
            prob = np.zeros((dim, num_bins), dtype=np.float64)
            indices = indices.reshape((num_bins, bin_size))
            for b, idx in enumerate(indices):
                basis_states, counts = np.unique(idx, return_counts=True)
                prob[basis_states, b] = counts / bin_size
            return prob
        prob = np.zeros((batch_size, dim, num_bins), dtype=np.float64)
        indices = indices.reshape((batch_size, num_bins, bin_size))
        for i, _indices in enumerate(indices):
            for b, idx in enumerate(_indices):
                basis_states, counts = np.unique(idx, return_counts=True)
                prob[i, basis_states, b] = counts / bin_size
        return prob

    def probability(self, wires=None, shot_range=None, bin_size=None):
        """Return either the analytic probability or estimated probability of
        each computational basis state.

        Devices that require a finite number of shots always return the
        estimated probability.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """
        wires = wires or self.wires
        if self.shots is None:
            return self.analytic_probability(wires=wires)
        return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

    @staticmethod
    def _get_batch_size(tensor, expected_shape, expected_size):
        """Determine whether a tensor has an additional batch dimension for broadcasting,
        compared to an expected_shape. As QubitDevice does not natively support broadcasting,
        it always reports no batch size, that is ``batch_size=None``"""
        return None

    def marginal_prob(self, prob, wires=None):
        """Return the marginal probability of the computational basis
        states by summing the probabiliites on the non-specified wires.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            If the provided wires are not in the order as they appear on the device,
            the returned marginal probabilities take this permutation into account.

            For example, if the addressable wires on this device are ``Wires([0, 1, 2])`` and
            this function gets passed ``wires=[2, 0]``, then the returned marginal
            probability vector will take this 'reversal' of the two wires
            into account:

            .. math::

                \\mathbb{P}^{(2, 0)}
                            = \\left[
                               |00\\rangle, |10\\rangle, |01\\rangle, |11\\rangle
                              \\right]

        Args:
            prob: The probabilities to return the marginal probabilities
                for
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            array[float]: array of the resulting marginal probabilities.
        """
        dim = 2 ** self.num_wires
        batch_size = self._get_batch_size(prob, (dim,), dim)
        if wires is None:
            return prob
        wires = Wires(wires)
        inactive_wires = Wires.unique_wires([self.wires, wires])
        device_wires = self.map_wires(wires)
        inactive_device_wires = self.map_wires(inactive_wires)
        if isinstance(inactive_device_wires, Wires):
            inactive_device_wires = inactive_device_wires.labels
        shape = [2] * self.num_wires
        desired_axes = np.argsort(np.argsort(device_wires))
        flat_shape = (-1,)
        if batch_size is not None:
            shape.insert(0, batch_size)
            inactive_device_wires = [idx + 1 for idx in inactive_device_wires]
            desired_axes = np.insert(desired_axes + 1, 0, 0)
            flat_shape = (batch_size, -1)
        prob = self._reshape(prob, shape)
        prob = self._reduce_sum(prob, inactive_device_wires)
        prob = self._transpose(prob, desired_axes)
        return self._reshape(prob, flat_shape)

    def expval(self, observable, shot_range=None, bin_size=None):
        if self.shots is None:
            try:
                eigvals = self._asarray(observable.eigvals() if not isinstance(observable, MeasurementValue) else [observable[i] for i in range(2 ** len(observable.measurements))], dtype=self.R_DTYPE)
            except qml.operation.EigvalsUndefinedError as e:
                raise qml.operation.EigvalsUndefinedError(f'Cannot compute analytic expectations of {observable.name}.') from e
            prob = self.probability(wires=observable.wires)
            return self._dot(prob, eigvals)
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        axis = -1 if bin_size is None else -2
        return np.squeeze(np.mean(samples, axis=axis))

    def var(self, observable, shot_range=None, bin_size=None):
        if self.shots is None:
            try:
                eigvals = self._asarray(observable.eigvals() if not isinstance(observable, MeasurementValue) else [observable[i] for i in range(2 ** len(observable.measurements))], dtype=self.R_DTYPE)
            except qml.operation.EigvalsUndefinedError as e:
                raise qml.operation.EigvalsUndefinedError(f'Cannot compute analytic variance of {observable.name}.') from e
            prob = self.probability(wires=observable.wires)
            return self._dot(prob, eigvals ** 2) - self._dot(prob, eigvals) ** 2
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        axis = -1 if bin_size is None else -2
        return np.squeeze(np.var(samples, axis=axis))

    def _samples_to_counts(self, samples, mp: CountsMP, num_wires):
        """Groups the samples into a dictionary showing number of occurences for
        each possible outcome.

        The format of the dictionary depends on mp.return_type, which is set when
        calling measurements.counts by setting the kwarg all_outcomes (bool). By default,
        the dictionary will only contain the observed outcomes. Optionally (all_outcomes=True)
        the dictionary will instead contain all possible outcomes, with a count of 0
        for those not observed. See example.


        Args:
            samples: An array of samples, with the shape being ``(shots,len(wires))`` if no observable
                is provided, with sample values being an array of 0s or 1s for each wire. Otherwise, it
                has shape ``(shots,)``, with sample values being scalar eigenvalues of the observable
            mp (~.measurements.CountsMP): the measurement process sampled
            num_wires (int): number of wires the sampled observable was performed on

        Returns:
            dict: dictionary with format ``{'outcome': num_occurences}``, including all
                outcomes for the sampled observable

        **Example**

            >>> num_wires = 2
            >>> dev = qml.device("default.qubit.legacy", wires=num_wires)
            >>> mp = qml.counts()
            >>> samples = pnp.array([[0, 0], [0, 0], [1, 0]])
            >>> dev._samples_to_counts(samples, mp, num_wires)
            {'00': 2, '10': 1}
            >>> mp = qml.counts(all_outcomes=True)
            >>> dev._samples_to_counts(samples, mp, num_wires)
            {'00': 2, '01': 0, '10': 1, '11': 0}

            The variable all_outcomes can be set when running measurements.counts, i.e.:

             .. code-block:: python3

                dev = qml.device("default.qubit", wires=2, shots=4)

                @qml.qnode(dev)
                def circuit(x):
                    qml.RX(x, wires=0)
                    return qml.counts(all_outcomes=True)

        """
        outcomes = []
        batched_ndims = 2
        shape = samples.shape
        if mp.obs is None and (not isinstance(mp.mv, MeasurementValue)):
            samples = np.array([sample for sample in samples if not np.any(np.isnan(sample))])
            samples = qml.math.cast_like(samples, qml.math.int8(0))
            samples = np.apply_along_axis(_sample_to_str, -1, samples)
            batched_ndims = 3
            if mp.all_outcomes:
                outcomes = list(map(_sample_to_str, self.generate_basis_states(num_wires)))
        elif mp.all_outcomes:
            outcomes = mp.eigvals()
        batched = len(shape) == batched_ndims
        if not batched:
            samples = samples[None]
        base_dict = {k: np.int64(0) for k in outcomes}
        outcome_dicts = [base_dict.copy() for _ in range(shape[0])]
        results = [np.unique(batch, return_counts=True) for batch in samples]
        for result, outcome_dict in zip(results, outcome_dicts):
            states, counts = result
            for state, count in zip(states, counts):
                outcome_dict[state] = count
        return outcome_dicts if batched else outcome_dicts[0]

    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        """Return samples of an observable.

        Args:
            observable (Observable): the observable to sample
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.
            counts (bool): whether counts (``True``) or raw samples (``False``)
                should be returned

        Raises:
            EigvalsUndefinedError: if no information is available about the
                eigenvalues of the observable

        Returns:
            Union[array[float], dict, list[dict]]: samples in an array of
            dimension ``(shots,)`` or counts
        """
        mp = observable
        no_observable_provided = False
        if isinstance(mp, MeasurementProcess):
            if mp.obs is not None:
                observable = mp.obs
            elif mp.mv is not None and isinstance(mp.mv, MeasurementValue):
                observable = mp.mv
            else:
                no_observable_provided = True
        device_wires = self.map_wires(observable.wires)
        name = None if no_observable_provided else observable.name
        if shot_range is None:
            sub_samples = self._samples
        else:
            sub_samples = self._samples[..., slice(*shot_range), :]
        if isinstance(name, str) and name in {'PauliX', 'PauliY', 'PauliZ', 'Hadamard'}:
            samples = 1 - 2 * sub_samples[..., device_wires[0]]
        elif no_observable_provided:
            if len(observable.wires) != 0:
                samples = sub_samples[..., np.array(device_wires)]
            else:
                samples = sub_samples
        else:
            samples = sub_samples[..., np.array(device_wires)]
            powers_of_two = 2 ** np.arange(samples.shape[-1])[::-1]
            indices = samples @ powers_of_two
            indices = np.array(indices)
            if isinstance(observable, MeasurementValue):
                eigvals = self._asarray([observable[i] for i in range(2 ** len(observable.measurements))], dtype=self.R_DTYPE)
                samples = eigvals[indices]
            else:
                try:
                    samples = observable.eigvals()[indices]
                except qml.operation.EigvalsUndefinedError as e:
                    raise qml.operation.EigvalsUndefinedError(f'Cannot compute samples of {observable.name}.') from e
        num_wires = len(device_wires) if len(device_wires) > 0 else self.num_wires
        if bin_size is None:
            if counts:
                return self._samples_to_counts(samples, mp, num_wires)
            return samples
        if counts:
            shape = (-1, bin_size, num_wires) if no_observable_provided else (-1, bin_size)
            return [self._samples_to_counts(bin_sample, mp, num_wires) for bin_sample in samples.reshape(shape)]
        return samples.T.reshape((num_wires, bin_size, -1)) if no_observable_provided else samples.reshape((bin_size, -1))

    def adjoint_jacobian(self, tape: QuantumTape, starting_state=None, use_device_state=False):
        """Implements the adjoint method outlined in
        `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

        After a forward pass, the circuit is reversed by iteratively applying adjoint
        gates to scan backwards through the circuit.

        .. note::
            The adjoint differentiation method has the following restrictions:

            * As it requires knowledge of the statevector, only statevector simulator devices can be
              used.

            * Only expectation values are supported as measurements.

            * Cannot differentiate with respect to state-prep operations.

            * Does not work for parametrized observables like
              :class:`~.Hamiltonian` or :class:`~.Hermitian`.

        Args:
            tape (.QuantumTape): circuit that the function takes the gradient of

        Keyword Args:
            starting_state (tensor_like): post-forward pass state to start execution with. It should be
                complex-valued. Takes precedence over ``use_device_state``.
            use_device_state (bool): use current device state to initialize. A forward pass of the same
                circuit should be the last thing the device has executed. If a ``starting_state`` is
                provided, that takes precedence.

        Returns:
            array or tuple[array]: the derivative of the tape with respect to trainable parameters.
            Dimensions are ``(len(observables), len(trainable_params))``.

        Raises:
            QuantumFunctionError: if the input tape has measurements that are not expectation values
                or contains a multi-parameter operation aside from :class:`~.Rot`
        """
        if tape.batch_size is not None:
            raise qml.QuantumFunctionError('Parameter broadcasting is not supported with adjoint differentiation')
        sum_axes = tuple(range(1, self.num_wires + 1))
        dot_product_real = lambda b, k: self._real(qmlsum(self._conj(b) * k, axis=sum_axes))
        for m in tape.measurements:
            if not isinstance(m, ExpectationMP):
                raise qml.QuantumFunctionError(f'Adjoint differentiation method does not support measurement {m.__class__.__name__}')
            if m.obs.name == 'Hamiltonian':
                raise qml.QuantumFunctionError('Adjoint differentiation method does not support Hamiltonian observables.')
        if self.shot_vector is not None:
            raise qml.QuantumFunctionError('Adjoint does not support shot vectors.')
        if self.shots is not None:
            warnings.warn('Requested adjoint differentiation to be computed with finite shots. The derivative is always exact when using the adjoint differentiation method.', UserWarning)
        if starting_state is not None:
            ket = self._reshape(starting_state, [2] * self.num_wires)
        else:
            if not use_device_state:
                self.reset()
                self.execute(tape)
            ket = self._pre_rotated_state
        n_obs = len(tape.observables)
        bras = np.empty([n_obs] + [2] * self.num_wires, dtype=np.complex128)
        for kk in range(n_obs):
            bras[kk, ...] = self._apply_operation(ket, tape.observables[kk])
        expanded_ops = []
        for op in reversed(tape.operations):
            if op.num_params > 1:
                if not isinstance(op, qml.Rot):
                    raise qml.QuantumFunctionError(f'The {op.name} operation is not supported using the "adjoint" differentiation method')
                ops = op.decomposition()
                expanded_ops.extend(reversed(ops))
            elif op.name not in ('StatePrep', 'QubitStateVector', 'BasisState', 'Snapshot'):
                expanded_ops.append(op)
        trainable_params = []
        for k in tape.trainable_params:
            mp_or_op = tape[tape._par_info[k]['op_idx']]
            if isinstance(mp_or_op, MeasurementProcess):
                warnings.warn(f'Differentiating with respect to the input parameters of {mp_or_op.obs.name} is not supported with the adjoint differentiation method. Gradients are computed only with regards to the trainable parameters of the circuit.\n\n Mark the parameters of the measured observables as non-trainable to silence this warning.', UserWarning)
            else:
                trainable_params.append(k)
        jac = np.zeros((len(tape.observables), len(trainable_params)))
        param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
        trainable_param_number = len(trainable_params) - 1
        for op in expanded_ops:
            adj_op = qml.adjoint(op)
            ket = self._apply_operation(ket, adj_op)
            if op.num_params == 1:
                if param_number in trainable_params:
                    d_op_matrix = operation_derivative(op)
                    ket_temp = self._apply_unitary(ket, d_op_matrix, op.wires)
                    jac[:, trainable_param_number] = 2 * dot_product_real(bras, ket_temp)
                    trainable_param_number -= 1
                param_number -= 1
            for kk in range(n_obs):
                bras[kk, ...] = self._apply_operation(bras[kk, ...], adj_op)
        return self._adjoint_jacobian_processing(jac)

    @staticmethod
    def _adjoint_jacobian_processing(jac):
        """
        Post-process the Jacobian matrix returned by ``adjoint_jacobian`` for
        the new return type system.
        """
        jac = np.squeeze(jac)
        if jac.ndim == 0:
            return np.array(jac)
        if jac.ndim == 1:
            return tuple((np.array(j) for j in jac))
        return tuple((tuple((np.array(j_) for j_ in j)) for j in jac))

    def _get_diagonalizing_gates(self, circuit: QuantumTape) -> List[Operation]:
        """Returns the gates that diagonalize the measured wires such that they
        are in the eigenbasis of the circuit observables.

        Note that this exists as a method of the Device class to enable child classes to
        override the implementation if necessary (for example, to skip computing rotation
        gates for a measurement that doesn't need them).

        Args:
            circuit (~.tape.QuantumTape): The circuit containing observables that may need diagonalizing

        Returns:
            List[~.Operation]: the operations that diagonalize the observables
        """
        return circuit.diagonalizing_gates