from __future__ import annotations
import uuid
import time
import logging
import warnings
from collections import Counter
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.providers import Provider
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.options import Options
from qiskit.qobj import QasmQobj, QasmQobjConfig, QasmQobjExperiment
from qiskit.result import Result
from qiskit.transpiler import Target
from .basic_provider_job import BasicProviderJob
from .basic_provider_tools import single_gate_matrix
from .basic_provider_tools import SINGLE_QUBIT_GATES
from .basic_provider_tools import cx_gate_matrix
from .basic_provider_tools import einsum_vecmul_index
from .exceptions import BasicProviderError
class BasicSimulator(BackendV2):
    """Python implementation of a basic (non-efficient) quantum simulator."""
    MAX_QUBITS_MEMORY = 24

    def __init__(self, provider: Provider | None=None, target: Target | None=None, **fields) -> None:
        """
        Args:
            provider: An optional backwards reference to the
                :class:`~qiskit.providers.Provider` object that the backend
                is from.
            target: An optional target to configure the simulator.
            fields: kwargs for the values to use to override the default
                options.

        Raises:
            AttributeError: If a field is specified that's outside the backend's
                options.
        """
        super().__init__(provider=provider, name='basic_simulator', description='A python simulator for quantum experiments', backend_version='0.1', **fields)
        self._target = target
        self._configuration = None
        self._local_random = None
        self._classical_memory = 0
        self._classical_register = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._memory = False
        self._initial_statevector = self.options.get('initial_statevector')
        self._chop_threshold = self.options.get('chop_threashold')
        self._qobj_config = None
        self._sample_measure = False

    @property
    def max_circuits(self) -> None:
        return None

    @property
    def target(self) -> Target:
        if not self._target:
            self._target = self._build_basic_target()
        return self._target

    def _build_basic_target(self) -> Target:
        """Helper method that returns a minimal target with a basis gate set but
        no coupling map, instruction properties or calibrations.

        Returns:
            The configured target.
        """
        target = Target(description='Basic Target', num_qubits=None)
        basis_gates = ['h', 'u', 'p', 'u1', 'u2', 'u3', 'rz', 'sx', 'x', 'cx', 'id', 'unitary', 'measure', 'delay', 'reset']
        inst_mapping = get_standard_gate_name_mapping()
        for name in basis_gates:
            if name in inst_mapping:
                instruction = inst_mapping[name]
                target.add_instruction(instruction, properties=None, name=name)
            elif name == 'unitary':
                target.add_instruction(UnitaryGate, name='unitary')
            else:
                raise BasicProviderError('Gate is not a valid basis gate for this simulator: %s' % name)
        return target

    def configuration(self) -> BackendConfiguration:
        """Return the simulator backend configuration.

        Returns:
            The configuration for the backend.
        """
        if self._configuration:
            return self._configuration
        gates = [{'name': name, 'parameters': self.target.operation_from_name(name).params} for name in self.target.operation_names]
        self._configuration = BackendConfiguration(backend_name=self.name, backend_version=self.backend_version, n_qubits=self.num_qubits, basis_gates=self.target.operation_names, gates=gates, local=True, simulator=True, conditional=True, open_pulse=False, memory=True, max_shots=0, coupling_map=None, description='A python simulator for quantum experiments')
        return self._configuration

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, memory=False, initial_statevector=None, chop_threshold=1e-15, allow_sample_measuring=True, seed_simulator=None, parameter_binds=None)

    def _add_unitary(self, gate: np.ndarray, qubits: list[int]) -> None:
        """Apply an N-qubit unitary matrix.

        Args:
            gate (matrix_like): an N-qubit unitary matrix
            qubits (list): the list of N-qubits.
        """
        num_qubits = len(qubits)
        indexes = einsum_vecmul_index(qubits, self._number_of_qubits)
        gate_tensor = np.reshape(np.array(gate, dtype=complex), num_qubits * [2, 2])
        self._statevector = np.einsum(indexes, gate_tensor, self._statevector, dtype=complex, casting='no')

    def _get_measure_outcome(self, qubit: int) -> tuple[str, int]:
        """Simulate the outcome of measurement of a qubit.

        Args:
            qubit: the qubit to measure

        Return:
            pair (outcome, probability) where outcome is '0' or '1' and
            probability is the probability of the returned outcome.
        """
        axis = list(range(self._number_of_qubits))
        axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis))
        random_number = self._local_random.random()
        if random_number < probabilities[0]:
            return ('0', probabilities[0])
        return ('1', probabilities[1])

    def _add_sample_measure(self, measure_params: list[list[int, int]], num_samples: int) -> list[hex]:
        """Generate memory samples from current statevector.

        Args:
            measure_params: List of (qubit, cmembit) values for
                                   measure instructions to sample.
            num_samples: The number of memory samples to generate.

        Returns:
            A list of memory values in hex format.
        """
        measured_qubits = sorted({qubit for qubit, cmembit in measure_params})
        num_measured = len(measured_qubits)
        axis = list(range(self._number_of_qubits))
        for qubit in reversed(measured_qubits):
            axis.remove(self._number_of_qubits - 1 - qubit)
        probabilities = np.reshape(np.sum(np.abs(self._statevector) ** 2, axis=tuple(axis)), 2 ** num_measured)
        samples = self._local_random.choice(range(2 ** num_measured), num_samples, p=probabilities)
        memory = []
        for sample in samples:
            classical_memory = self._classical_memory
            for qubit, cmembit in measure_params:
                pos = measured_qubits.index(qubit)
                qubit_outcome = int((sample & 1 << pos) >> pos)
                membit = 1 << cmembit
                classical_memory = classical_memory & ~membit | qubit_outcome << cmembit
            value = bin(classical_memory)[2:]
            memory.append(hex(int(value, 2)))
        return memory

    def _add_measure(self, qubit: int, cmembit: int, cregbit: int | None=None) -> None:
        """Apply a measure instruction to a qubit.

        Args:
            qubit: qubit is the qubit measured.
            cmembit: is the classical memory bit to store outcome in.
            cregbit: is the classical register bit to store outcome in.
        """
        outcome, probability = self._get_measure_outcome(qubit)
        membit = 1 << cmembit
        self._classical_memory = self._classical_memory & ~membit | int(outcome) << cmembit
        if cregbit is not None:
            regbit = 1 << cregbit
            self._classical_register = self._classical_register & ~regbit | int(outcome) << cregbit
        if outcome == '0':
            update_diag = [[1 / np.sqrt(probability), 0], [0, 0]]
        else:
            update_diag = [[0, 0], [0, 1 / np.sqrt(probability)]]
        self._add_unitary(update_diag, [qubit])

    def _add_reset(self, qubit: int) -> None:
        """Apply a reset instruction to a qubit.

        Args:
            qubit: the qubit being rest

        This is done by doing a simulating a measurement
        outcome and projecting onto the outcome state while
        renormalizing.
        """
        outcome, probability = self._get_measure_outcome(qubit)
        if outcome == '0':
            update = [[1 / np.sqrt(probability), 0], [0, 0]]
            self._add_unitary(update, [qubit])
        else:
            update = [[0, 1 / np.sqrt(probability)], [0, 0]]
            self._add_unitary(update, [qubit])

    def _validate_initial_statevector(self) -> None:
        """Validate an initial statevector"""
        if self._initial_statevector is None:
            return
        length = len(self._initial_statevector)
        required_dim = 2 ** self._number_of_qubits
        if length != required_dim:
            raise BasicProviderError(f'initial statevector is incorrect length: {length} != {required_dim}')

    def _set_options(self, qobj_config: QasmQobjConfig | None=None, backend_options: dict | None=None) -> None:
        """Set the backend options for all experiments in a qobj"""
        self._initial_statevector = self.options.get('initial_statevector')
        self._chop_threshold = self.options.get('chop_threshold')
        if 'backend_options' in backend_options and backend_options['backend_options']:
            backend_options = backend_options['backend_options']
        if 'initial_statevector' in backend_options and backend_options['initial_statevector'] is not None:
            self._initial_statevector = np.array(backend_options['initial_statevector'], dtype=complex)
        elif hasattr(qobj_config, 'initial_statevector'):
            self._initial_statevector = np.array(qobj_config.initial_statevector, dtype=complex)
        if self._initial_statevector is not None:
            norm = np.linalg.norm(self._initial_statevector)
            if round(norm, 12) != 1:
                raise BasicProviderError(f'initial statevector is not normalized: norm {norm} != 1')
        if 'chop_threshold' in backend_options:
            self._chop_threshold = backend_options['chop_threshold']
        elif hasattr(qobj_config, 'chop_threshold'):
            self._chop_threshold = qobj_config.chop_threshold

    def _initialize_statevector(self) -> None:
        """Set the initial statevector for simulation"""
        if self._initial_statevector is None:
            self._statevector = np.zeros(2 ** self._number_of_qubits, dtype=complex)
            self._statevector[0] = 1
        else:
            self._statevector = self._initial_statevector.copy()
        self._statevector = np.reshape(self._statevector, self._number_of_qubits * [2])

    def _validate_measure_sampling(self, experiment: QasmQobjExperiment) -> None:
        """Determine if measure sampling is allowed for an experiment

        Args:
            experiment: a qobj experiment.
        """
        if self._shots <= 1:
            self._sample_measure = False
            return
        if hasattr(experiment.config, 'allows_measure_sampling'):
            self._sample_measure = experiment.config.allows_measure_sampling
        else:
            measure_flag = False
            for instruction in experiment.instructions:
                if instruction.name == 'reset':
                    self._sample_measure = False
                    return
                if measure_flag:
                    if instruction.name not in ['measure', 'barrier', 'id', 'u0']:
                        self._sample_measure = False
                        return
                elif instruction.name == 'measure':
                    measure_flag = True
            self._sample_measure = True

    def run(self, run_input: QuantumCircuit | list[QuantumCircuit], **backend_options) -> BasicProviderJob:
        """Run on the backend.

        Args:
            run_input: payload of the experiment
            backend_options: backend options

        Returns:
            BasicProviderJob: derived from BaseJob

        Additional Information:
            backend_options: Is a dict of options for the backend. It may contain
                * "initial_statevector": vector_like

            The "initial_statevector" option specifies a custom initial
            initial statevector for the simulator to be used instead of the all
            zero state. This size of this vector must be correct for the number
            of qubits in ``run_input`` parameter.

            Example::

                backend_options = {
                    "initial_statevector": np.array([1, 0, 0, 1j]) / np.sqrt(2),
                }
        """
        from qiskit.compiler import assemble
        out_options = {}
        for key in backend_options:
            if not hasattr(self.options, key):
                warnings.warn('Option %s is not used by this backend' % key, UserWarning, stacklevel=2)
            else:
                out_options[key] = backend_options[key]
        qobj = assemble(run_input, self, **out_options)
        qobj_options = qobj.config
        self._set_options(qobj_config=qobj_options, backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = BasicProviderJob(self, job_id, self._run_job(job_id, qobj))
        return job

    def _run_job(self, job_id: str, qobj: QasmQobj) -> Result:
        """Run experiments in qobj

        Args:
            job_id: unique id for the job.
            qobj: job description

        Returns:
            Result object
        """
        self._validate(qobj)
        result_list = []
        self._shots = qobj.config.shots
        self._memory = getattr(qobj.config, 'memory', False)
        self._qobj_config = qobj.config
        start = time.time()
        for experiment in qobj.experiments:
            result_list.append(self.run_experiment(experiment))
        end = time.time()
        result = {'backend_name': self.name, 'backend_version': self.backend_version, 'qobj_id': qobj.qobj_id, 'job_id': job_id, 'results': result_list, 'status': 'COMPLETED', 'success': True, 'time_taken': end - start, 'header': qobj.header.to_dict()}
        return Result.from_dict(result)

    def run_experiment(self, experiment: QasmQobjExperiment) -> dict[str, ...]:
        """Run an experiment (circuit) and return a single experiment result.

        Args:
            experiment: experiment from qobj experiments list

        Returns:
             A result dictionary which looks something like::

                {
                "name": name of this experiment (obtained from qobj.experiment header)
                "seed": random seed used for simulation
                "shots": number of shots used in the simulation
                "data":
                    {
                    "counts": {'0x9: 5, ...},
                    "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                    },
                "status": status string for the simulation
                "success": boolean
                "time_taken": simulation time of this single experiment
                }
        Raises:
            BasicProviderError: if an error occurred.
        """
        start = time.time()
        self._number_of_qubits = experiment.config.n_qubits
        self._number_of_cmembits = experiment.config.memory_slots
        self._statevector = 0
        self._classical_memory = 0
        self._classical_register = 0
        self._sample_measure = False
        global_phase = experiment.header.global_phase
        self._validate_initial_statevector()
        if hasattr(experiment.config, 'seed_simulator'):
            seed_simulator = experiment.config.seed_simulator
        elif hasattr(self._qobj_config, 'seed_simulator'):
            seed_simulator = self._qobj_config.seed_simulator
        else:
            seed_simulator = np.random.randint(2147483647, dtype='int32')
        self._local_random = np.random.default_rng(seed=seed_simulator)
        self._validate_measure_sampling(experiment)
        memory = []
        if self._sample_measure:
            shots = 1
            measure_sample_ops = []
        else:
            shots = self._shots
        for _ in range(shots):
            self._initialize_statevector()
            self._statevector *= np.exp(1j * global_phase)
            self._classical_memory = 0
            self._classical_register = 0
            for operation in experiment.instructions:
                conditional = getattr(operation, 'conditional', None)
                if isinstance(conditional, int):
                    conditional_bit_set = self._classical_register >> conditional & 1
                    if not conditional_bit_set:
                        continue
                elif conditional is not None:
                    mask = int(operation.conditional.mask, 16)
                    if mask > 0:
                        value = self._classical_memory & mask
                        while mask & 1 == 0:
                            mask >>= 1
                            value >>= 1
                        if value != int(operation.conditional.val, 16):
                            continue
                if operation.name == 'unitary':
                    qubits = operation.qubits
                    gate = operation.params[0]
                    self._add_unitary(gate, qubits)
                elif operation.name in SINGLE_QUBIT_GATES:
                    params = getattr(operation, 'params', None)
                    qubit = operation.qubits[0]
                    gate = single_gate_matrix(operation.name, params)
                    self._add_unitary(gate, [qubit])
                elif operation.name in ('id', 'u0'):
                    pass
                elif operation.name in ('CX', 'cx'):
                    qubit0 = operation.qubits[0]
                    qubit1 = operation.qubits[1]
                    gate = cx_gate_matrix()
                    self._add_unitary(gate, [qubit0, qubit1])
                elif operation.name == 'reset':
                    qubit = operation.qubits[0]
                    self._add_reset(qubit)
                elif operation.name == 'barrier':
                    pass
                elif operation.name == 'measure':
                    qubit = operation.qubits[0]
                    cmembit = operation.memory[0]
                    cregbit = operation.register[0] if hasattr(operation, 'register') else None
                    if self._sample_measure:
                        measure_sample_ops.append((qubit, cmembit))
                    else:
                        self._add_measure(qubit, cmembit, cregbit)
                elif operation.name == 'bfunc':
                    mask = int(operation.mask, 16)
                    relation = operation.relation
                    val = int(operation.val, 16)
                    cregbit = operation.register
                    cmembit = operation.memory if hasattr(operation, 'memory') else None
                    compared = (self._classical_register & mask) - val
                    if relation == '==':
                        outcome = compared == 0
                    elif relation == '!=':
                        outcome = compared != 0
                    elif relation == '<':
                        outcome = compared < 0
                    elif relation == '<=':
                        outcome = compared <= 0
                    elif relation == '>':
                        outcome = compared > 0
                    elif relation == '>=':
                        outcome = compared >= 0
                    else:
                        raise BasicProviderError('Invalid boolean function relation.')
                    regbit = 1 << cregbit
                    self._classical_register = self._classical_register & ~regbit | int(outcome) << cregbit
                    if cmembit is not None:
                        membit = 1 << cmembit
                        self._classical_memory = self._classical_memory & ~membit | int(outcome) << cmembit
                else:
                    backend = self.name
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise BasicProviderError(err_msg.format(backend, operation.name))
            if self._number_of_cmembits > 0:
                if self._sample_measure:
                    memory = self._add_sample_measure(measure_sample_ops, self._shots)
                else:
                    outcome = bin(self._classical_memory)[2:]
                    memory.append(hex(int(outcome, 2)))
        data = {'counts': dict(Counter(memory))}
        if self._memory:
            data['memory'] = memory
        end = time.time()
        return {'name': experiment.header.name, 'seed_simulator': seed_simulator, 'shots': self._shots, 'data': data, 'status': 'DONE', 'success': True, 'time_taken': end - start, 'header': experiment.header.to_dict()}

    def _validate(self, qobj: QasmQobj) -> None:
        """Semantic validations of the qobj which cannot be done via schemas."""
        n_qubits = qobj.config.n_qubits
        max_qubits = self.MAX_QUBITS_MEMORY
        if n_qubits > max_qubits:
            raise BasicProviderError(f'Number of qubits {n_qubits} is greater than maximum ({max_qubits}) for "{self.name}".')
        for experiment in qobj.experiments:
            name = experiment.header.name
            if experiment.config.memory_slots == 0:
                logger.warning('No classical registers in circuit "%s", counts will be empty.', name)
            elif 'measure' not in [op.name for op in experiment.instructions]:
                logger.warning('No measurements in circuit "%s", classical register will remain all zeros.', name)