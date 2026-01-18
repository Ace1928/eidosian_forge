import itertools
import warnings
from math import log, pi
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._compiler import AbstractCompiler, QVMCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3
from pyquil.api._quantum_computer import get_qc as get_qc_v3, QuantumExecutable
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilatom import qubit_index
from ._qam import StatefulQAM
class QuantumComputer(QuantumComputerV3):
    compiler: AbstractCompiler
    qam: StatefulQAM[Any]

    def __init__(self, *, name: str, qam: QAM[Any], device: Any=None, compiler: AbstractCompiler, symmetrize_readout: bool=False) -> None:
        """
        An interface designed to ease migration from pyQuil v2 to v3, and compatible with most
        use cases for the pyQuil v2 QuantumComputer.

        A quantum computer for running quantum programs.

        A quantum computer has various characteristics like supported gates, qubits, qubit
        topologies, gate fidelities, and more. A quantum computer also has the ability to
        run quantum programs.

        A quantum computer can be a real Rigetti QPU that uses superconducting transmon
        qubits to run quantum programs, or it can be an emulator like the QVM with
        noise models and mimicked topologies.

        :param name: A string identifying this particular quantum computer.
        :param qam: A quantum abstract machine which handles executing quantum programs. This
            dispatches to a QVM or QPU.
        :param device: Ignored and accepted only for backwards compatibility.
        :param symmetrize_readout: Whether to apply readout error symmetrization. See
            :py:func:`run_symmetrized_readout` for a complete description.
        """
        self.name = name
        StatefulQAM.wrap(qam)
        self.qam = cast(StatefulQAM[Any], qam)
        self.compiler = compiler
        self.symmetrize_readout = symmetrize_readout

    def run(self, executable: QuantumExecutable, memory_map: Optional[Mapping[str, Sequence[Union[int, float]]]]=None) -> np.ndarray:
        """
        Run a quil executable. If the executable contains declared parameters, then a memory
        map must be provided, which defines the runtime values of these parameters.

        :param executable: The program to run. You are responsible for compiling this first.
        :param memory_map: The mapping of declared parameters to their values. The values
            are a list of floats or integers.
        :return: A numpy array of shape (trials, len(ro-register)) that contains 0s and 1s.
        """
        self.qam.load(executable)
        if memory_map:
            for region_name, values_list in memory_map.items():
                self.qam.write_memory(region_name=region_name, value=values_list)
        result = self.qam.run().read_memory(region_name='ro')
        assert result is not None
        return result

    def calibrate(self, experiment: Experiment) -> List[ExperimentResult]:
        """
        Perform readout calibration on the various multi-qubit observables involved in the provided
        ``Experiment``.

        :param experiment: The ``Experiment`` to calibrate readout error for.
        :return: A list of ``ExperimentResult`` objects that contain the expectation values that
            correspond to the scale factors resulting from symmetric readout error.
        """
        calibration_experiment = experiment.generate_calibration_experiment()
        return self.experiment(calibration_experiment)

    def experiment(self, experiment: Experiment, memory_map: Optional[Mapping[str, Sequence[Union[int, float]]]]=None) -> List[ExperimentResult]:
        """
        Run an ``Experiment`` on a QVM or QPU backend. An ``Experiment`` is composed of:

            - A main ``Program`` body (or ansatz).
            - A collection of ``ExperimentSetting`` objects, each of which encodes a particular
              state preparation and measurement.
            - A ``SymmetrizationLevel`` for enacting different readout symmetrization strategies.
            - A number of shots to collect for each (unsymmetrized) ``ExperimentSetting``.

        Because the main ``Program`` is static from run to run of an ``Experiment``, we can leverage
        our platform's Parametric Compilation feature. This means that the ``Program`` can be
        compiled only once, and the various alterations due to state preparation, measurement,
        and symmetrization can all be realized at runtime by providing a ``memory_map``. Thus, the
        steps in the ``experiment`` method are as follows:

            1. Generate a parameterized program corresponding to the ``Experiment``
                (see the ``Experiment.generate_experiment_program()`` method for more
                details on how it changes the main body program to support state preparation,
                measurement, and symmetrization).
            2. Compile the parameterized program into a parametric (binary) executable, which
                   contains declared variables that can be assigned at runtime.

            3. For each ``ExperimentSetting`` in the ``Experiment``, we repeat the following:

                a. Build a collection of memory maps that correspond to the various state
                   preparation, measurement, and symmetrization specifications.
                b. Run the parametric executable on the QVM or QPU backend, providing the memory map
                   to assign variables at runtime.
                c. Extract the desired statistics from the classified bitstrings that are produced
                   by the QVM or QPU backend, and package them in an ``ExperimentResult`` object.

            4. Return the list of ``ExperimentResult`` objects.

        This method is extremely useful shorthand for running near-term applications and algorithms,
        which often have this ansatz + settings structure.

        :param experiment: The ``Experiment`` to run.
        :param memory_map: A dictionary mapping declared variables / parameters to their values.
            The values are a list of floats or integers. Each float or integer corresponds to
            a particular classical memory register. The memory map provided to the ``experiment``
            method corresponds to variables in the main body program that we would like to change
            at runtime (e.g. the variational parameters provided to the ansatz of the variational
            quantum eigensolver).
        :return: A list of ``ExperimentResult`` objects containing the statistics gathered
            according to the specifications of the ``Experiment``.
        """
        experiment_program = experiment.generate_experiment_program()
        executable = self.compile(experiment_program)
        if memory_map is None:
            memory_map = {}
        results = []
        for settings in experiment:
            if len(settings) > 1:
                raise ValueError('settings must be of length 1')
            setting = settings[0]
            qubits = cast(List[int], setting.out_operator.get_qubits())
            experiment_setting_memory_map = experiment.build_setting_memory_map(setting)
            symmetrization_memory_maps = experiment.build_symmetrization_memory_maps(qubits)
            merged_memory_maps = merge_memory_map_lists([experiment_setting_memory_map], symmetrization_memory_maps)
            all_bitstrings = []
            for merged_memory_map in merged_memory_maps:
                final_memory_map = {**memory_map, **merged_memory_map}
                self.qam.reset()
                bitstrings = self.run(executable, memory_map=final_memory_map)
                if 'symmetrization' in final_memory_map:
                    bitmask = np.array(np.array(final_memory_map['symmetrization']) / np.pi, dtype=int)
                    bitstrings = np.bitwise_xor(bitstrings, bitmask)
                all_bitstrings.append(bitstrings)
            symmetrized_bitstrings = np.concatenate(all_bitstrings)
            joint_expectations = [experiment.get_meas_registers(qubits)]
            if setting.additional_expectations:
                joint_expectations += setting.additional_expectations
            expectations = bitstrings_to_expectations(symmetrized_bitstrings, joint_expectations=joint_expectations)
            means = cast(np.ndarray, np.mean(expectations, axis=0))
            std_errs = np.std(expectations, axis=0, ddof=1) / np.sqrt(len(expectations))
            joint_results = []
            for qubit_subset, mean, std_err in zip(joint_expectations, means, std_errs):
                out_operator = PauliTerm.from_list([(setting.out_operator[i], i) for i in qubit_subset])
                s = ExperimentSetting(in_state=setting.in_state, out_operator=out_operator, additional_expectations=None)
                r = ExperimentResult(setting=s, expectation=mean, std_err=std_err, total_counts=len(expectations))
                joint_results.append(r)
            result = ExperimentResult(setting=setting, expectation=joint_results[0].expectation, std_err=joint_results[0].std_err, total_counts=joint_results[0].total_counts, additional_results=joint_results[1:])
            results.append(result)
        return results

    def run_symmetrized_readout(self, program: Program, trials: int, symm_type: int=3, meas_qubits: Optional[List[int]]=None) -> np.ndarray:
        """
        Run a quil program in such a way that the readout error is made symmetric. Enforcing
        symmetric readout error is useful in simplifying the assumptions in some near
        term error mitigation strategies, see ``measure_observables`` for more information.

        The simplest example is for one qubit. In a noisy device, the probability of accurately
        reading the 0 state might be higher than that of the 1 state; due to e.g. amplitude
        damping. This makes correcting for readout more difficult. In the simplest case, this
        function runs the program normally ``(trials//2)`` times. The other half of the time,
        it will insert an ``X`` gate prior to any ``MEASURE`` instruction and then flip the
        measured classical bit back. Overall this has the effect of symmetrizing the readout error.

        The details. Consider preparing the input bitstring ``|i>`` (in the computational basis) and
        measuring in the Z basis. Then the Confusion matrix for the readout error is specified by
        the probabilities

             p(j|i) := Pr(measured = j | prepared = i ).

        In the case of a single qubit i,j \\in [0,1] then:
        there is no readout error if p(0|0) = p(1|1) = 1.
        the readout error is symmetric if p(0|0) = p(1|1) = 1 - epsilon.
        the readout error is asymmetric if p(0|0) != p(1|1).

        If your quantum computer has this kind of asymmetric readout error then
        ``qc.run_symmetrized_readout`` will symmetrize the readout error.

        The readout error above is only asymmetric on a single bit. In practice the confusion
        matrix on n bits need not be symmetric, e.g. for two qubits p(ij|ij) != 1 - epsilon for
        all i,j. In these situations a more sophisticated means of symmetrization is needed; and
        we use orthogonal arrays (OA) built from Hadamard matrices.

        The symmetrization types are specified by an int; the types available are:
        -1 -- exhaustive symmetrization uses every possible combination of flips
        0 -- trivial that is no symmetrization
        1 -- symmetrization using an OA with strength 1
        2 -- symmetrization using an OA with strength 2
        3 -- symmetrization using an OA with strength 3
        In the context of readout symmetrization the strength of the orthogonal array enforces
        the symmetry of the marginal confusion matrices.

        By default a strength 3 OA is used; this ensures expectations of the form
        ``<b_k . b_j . b_i>`` for bits any bits i,j,k will have symmetric readout errors. Here
        expectation of a random variable x as is denote ``<x> = sum_i Pr(i) x_i``. It turns out that
        a strength 3 OA is also a strength 2 and strength 1 OA it also ensures ``<b_j . b_i>`` and
        ``<b_i>`` have symmetric readout errors for any bits b_j and b_i.

        :param program: The program to run symmetrized readout on.
        :param trials: The minimum number of times to run the program; it is recommend that this
            number should be in the hundreds or thousands. This parameter will be mutated if
            necessary.
        :param symm_type: the type of symmetrization
        :param meas_qubits: An advanced feature. The groups of measurement qubits. Only these
            qubits will be symmetrized over, even if the program acts on other qubits.
        :return: A numpy array of shape (trials, len(ro-register)) that contains 0s and 1s.
        """
        if not isinstance(symm_type, int):
            raise ValueError('Symmetrization options are indicated by an int. See the docstrings for more information.')
        if meas_qubits is None:
            meas_qubits = list(cast(Set[int], program.get_qubits()))
        trials = _check_min_num_trials_for_symmetrized_readout(len(meas_qubits), trials, symm_type)
        sym_programs, flip_arrays = _symmetrization(program, meas_qubits, symm_type)
        num_shots_per_prog = trials // len(sym_programs)
        if num_shots_per_prog * len(sym_programs) < trials:
            warnings.warn(f'The number of trials was modified from {trials} to {num_shots_per_prog * len(sym_programs)}. To be consistent with the number of trials required by the type of readout symmetrization chosen.')
        results = _measure_bitstrings(self, sym_programs, meas_qubits, num_shots_per_prog)
        return _consolidate_symmetrization_outputs(results, flip_arrays)

    def run_and_measure(self, program: Program, trials: int) -> Dict[int, np.ndarray]:
        """
        Run the provided state preparation program and measure all qubits.

        The returned data is a dictionary keyed by qubit index because qubits for a given
        QuantumComputer may be non-contiguous and non-zero-indexed. To turn this dictionary
        into a 2d numpy array of bitstrings, consider::

            bitstrings = qc.run_and_measure(...)
            bitstring_array = np.vstack([bitstrings[q] for q in qc.qubits()]).T
            bitstring_array.shape  # (trials, len(qc.qubits()))

        .. note::

            If the target :py:class:`QuantumComputer` is a noiseless :py:class:`QVM` then
            only the qubits explicitly used in the program will be measured. Otherwise all
            qubits will be measured. In some circumstances this can exhaust the memory
            available to the simulator, and this may be manifested by the QVM failing to
            respond or timeout.

        .. note::

            In contrast to :py:class:`QVMConnection.run_and_measure`, this method simulates
            noise correctly for noisy QVMs. However, this method is slower for ``trials > 1``.
            For faster noise-free simulation, consider
            :py:class:`WavefunctionSimulator.run_and_measure`.

        :param program: The state preparation program to run and then measure.
        :param trials: The number of times to run the program.
        :return: A dictionary keyed by qubit index where the corresponding value is a 1D array of
            measured bits.
        """
        program = program.copy()
        validate_supported_quil(program)
        ro = program.declare('ro', 'BIT', len(self.qubits()))
        measure_used = isinstance(self.qam, QVM) and self.qam.noise_model is None
        qubits_to_measure = set(map(qubit_index, program.get_qubits()) if measure_used else self.qubits())
        for i, q in enumerate(qubits_to_measure):
            program.inst(MEASURE(q, ro[i]))
        program.wrap_in_numshots_loop(trials)
        executable = self.compile(program)
        bitstring_array = self.run(executable=executable)
        bitstring_dict = {}
        for i, q in enumerate(qubits_to_measure):
            bitstring_dict[q] = bitstring_array[:, i]
        for q in set(self.qubits()) - set(qubits_to_measure):
            bitstring_dict[q] = np.zeros(trials)
        return bitstring_dict

    def compile(self, program: Program, to_native_gates: bool=True, optimize: bool=True, *, protoquil: Optional[bool]=None) -> QuantumExecutable:
        """
        A high-level interface to program compilation.

        Compilation currently consists of two stages. Please see the :py:class:`AbstractCompiler`
        docs for more information. This function does all stages of compilation.

        Right now both ``to_native_gates`` and ``optimize`` must be either both set or both
        unset. More modular compilation passes may be available in the future.

        Additionally, a call to compile also calls the ``reset`` method if one is running
        on the QPU. This is a bit of a sneaky hack to guard against stale compiler connections,
        but shouldn't result in any material hit to performance (especially when taking advantage
        of parametric compilation for hybrid applications).

        :param program: A Program
        :param to_native_gates: Whether to compile non-native gates to native gates.
        :param optimize: Whether to optimize the program to reduce the number of operations.
        :param protoquil: Whether to restrict the input program to and the compiled program
            to protoquil (executable on QPU). A value of ``None`` means defer to server.
        :return: An executable binary suitable for passing to :py:func:`QuantumComputer.run`.
        """
        if isinstance(self.qam, QPU):
            self.reset()
        flags = [to_native_gates, optimize]
        assert all(flags) or all((not f for f in flags)), 'Must turn quilc all on or all off'
        quilc = all(flags)
        if quilc:
            nq_program = self.compiler.quil_to_native_quil(program, protoquil=protoquil)
        else:
            nq_program = program
        binary = self.compiler.native_quil_to_executable(nq_program)
        return binary

    def reset(self) -> None:
        """
        Reset the QuantumComputer's QAM to its initial state.
        """
        self.qam.reset()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'QuantumComputer[name="{self.name}"]'