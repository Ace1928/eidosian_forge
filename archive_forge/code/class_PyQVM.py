import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.api import QAM, QuantumExecutable, QAMExecutionResult
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.quilatom import Label, LabelPlaceholder, MemoryReference
from pyquil.quilbase import (
class PyQVM(QAM['PyQVM']):

    def __init__(self, n_qubits: int, quantum_simulator_type: Optional[Type[AbstractQuantumSimulator]]=None, seed: Optional[int]=None, post_gate_noise_probabilities: Optional[Dict[str, float]]=None):
        """
        PyQuil's built-in Quil virtual machine.

        This class implements common control flow and plumbing and dispatches the "actual" work to
        quantum simulators like ReferenceWavefunctionSimulator, ReferenceDensitySimulator,
        and NumpyWavefunctionSimulator

        :param n_qubits: The number of qubits. Typically this results in the allocation of a large
            ndarray, so be judicious.
        :param quantum_simulator_type: A class that can be instantiated to handle the quantum
            aspects of this QVM. If not specified, the default will be either
            NumpyWavefunctionSimulator (no noise) or ReferenceDensitySimulator (noise)
        :param post_gate_noise_probabilities: A specification of noise model given by
            probabilities of certain types of noise. The dictionary keys are from "relaxation",
            "dephasing", "depolarizing", "phase_flip", "bit_flip", and "bitphase_flip".
            WARNING: experimental. This interface will likely change.
        :param seed: An optional random seed for performing stochastic aspects of the QVM.
        """
        if quantum_simulator_type is None:
            if post_gate_noise_probabilities is None:
                from pyquil.simulation._numpy import NumpyWavefunctionSimulator
                quantum_simulator_type = NumpyWavefunctionSimulator
            else:
                from pyquil.simulation._reference import ReferenceDensitySimulator
                log.info('Using ReferenceDensitySimulator as the backend for PyQVM')
                quantum_simulator_type = ReferenceDensitySimulator
        self.n_qubits = n_qubits
        self.ram: Dict[str, np.ndarray] = {}
        if post_gate_noise_probabilities is None:
            post_gate_noise_probabilities = {}
        self.post_gate_noise_probabilities = post_gate_noise_probabilities
        self.program: Optional[Program] = None
        self.program_counter: int = 0
        self.defined_gates: Dict[str, np.ndarray] = dict()
        self._qubit_to_ram: Optional[Dict[int, int]] = None
        self._ro_size: Optional[int] = None
        self._memory_results = {}
        self.rs = np.random.RandomState(seed=seed)
        self.wf_simulator = quantum_simulator_type(n_qubits=n_qubits, rs=self.rs)
        self._last_measure_program_loc = None

    def _extract_defined_gates(self) -> None:
        self.defined_gates = dict()
        assert self.program is not None
        for dg in self.program.defined_gates:
            if dg.parameters is not None and len(dg.parameters) > 0:
                raise NotImplementedError('PyQVM does not support parameterized DEFGATEs')
            if isinstance(dg, DefPermutationGate) or isinstance(dg, DefGateByPaulis):
                raise NotImplementedError('PyQVM does not support DEFGATE ... AS MATRIX | PAULI-SUM.')
            self.defined_gates[dg.name] = dg.matrix

    def write_memory(self, *, region_name: str, offset: int=0, value: int=0) -> 'PyQVM':
        assert region_name != 'ro'
        self.ram[region_name][offset] = value
        return self

    def execute(self, executable: QuantumExecutable) -> 'PyQVM':
        """
        Execute a program on the PyQVM. Note that the state of the instance is reset on each
        call to ``execute``.

        :return: ``self`` to support method chaining.
        """
        if not isinstance(executable, Program):
            raise TypeError('`executable` argument must be a `Program`')
        self.program = executable
        self._memory_results = {}
        self.ram = {}
        self.wf_simulator.reset()
        self._extract_defined_gates()
        self._memory_results = {}
        for _ in range(self.program.num_shots):
            self.wf_simulator.reset()
            self._execute_program()
            for name in self.ram.keys():
                self._memory_results.setdefault(name, list())
                self._memory_results[name].append(self.ram[name])
        self._memory_results = {k: np.asarray(v) for k, v in self._memory_results.items()}
        self._bitstrings = self._memory_results.get('ro')
        return self

    def get_result(self, execute_response: 'PyQVM') -> QAMExecutionResult:
        """
        Return results from the PyQVM according to the common QAM API. Note that while the
        ``execute_response`` is not used, it's accepted in order to conform to that API; it's
        unused because the PyQVM, unlike other QAM's, is itself stateful.
        """
        assert self.program is not None
        return QAMExecutionResult(executable=self.program.copy(), readout_data={k: v for k, v in self._memory_results.items()})

    def read_memory(self, *, region_name: str) -> np.ndarray:
        assert self._memory_results is not None
        return np.asarray(self._memory_results[region_name])

    def find_label(self, label: Union[Label, LabelPlaceholder]) -> int:
        """
        Helper function that iterates over the program and looks for a JumpTarget that has a
        Label matching the input label.

        :param label: Label object to search for in program
        :return: Program index where ``label`` is found
        """
        assert self.program is not None
        for index, action in enumerate(self.program):
            if isinstance(action, JumpTarget):
                if label == action.label:
                    return index
        raise RuntimeError('Improper program - Jump Target not found in the input program!')

    def transition(self) -> bool:
        """
        Implements a QAM-like transition.

        This function assumes ``program`` and ``program_counter`` instance variables are set
        appropriately, and that the wavefunction simulator and classical memory ``ram`` instance
        variables are in the desired QAM input state.

        :return: whether the QAM should halt after this transition.
        """
        assert self.program is not None
        instruction = self.program[self.program_counter]
        if isinstance(instruction, Gate):
            if instruction.name in self.defined_gates:
                self.wf_simulator.do_gate_matrix(matrix=self.defined_gates[instruction.name], qubits=[q.index for q in instruction.qubits])
            else:
                self.wf_simulator.do_gate(gate=instruction)
            for noise_type, noise_prob in self.post_gate_noise_probabilities.items():
                self.wf_simulator.do_post_gate_noise(noise_type, noise_prob, qubits=[q.index for q in instruction.qubits])
            self.program_counter += 1
        elif isinstance(instruction, Measurement):
            measured_val = self.wf_simulator.do_measurement(qubit=instruction.qubit.index)
            meas_reg: Optional[MemoryReference] = instruction.classical_reg
            assert meas_reg is not None
            self.ram[meas_reg.name][meas_reg.offset] = measured_val
            self.program_counter += 1
        elif isinstance(instruction, Declare):
            if instruction.shared_region is not None:
                raise NotImplementedError('SHARING is not (yet) implemented.')
            self.ram[instruction.name] = np.zeros(instruction.memory_size, dtype=QUIL_TO_NUMPY_DTYPE[instruction.memory_type])
            self.program_counter += 1
        elif isinstance(instruction, Pragma):
            self.program_counter += 1
        elif isinstance(instruction, Jump):
            self.program_counter = self.find_label(instruction.target)
        elif isinstance(instruction, JumpTarget):
            self.program_counter += 1
        elif isinstance(instruction, JumpConditional):
            jump_reg: Optional[MemoryReference] = instruction.condition
            assert jump_reg is not None
            cond = self.ram[jump_reg.name][jump_reg.offset]
            if not isinstance(cond, (bool, np.bool_, np.int8)):
                raise ValueError('{} requires a data type of BIT; not {}'.format(instruction.op, type(cond)))
            dest_index = self.find_label(instruction.target)
            if isinstance(instruction, JumpWhen):
                jump_if_cond = True
            elif isinstance(instruction, JumpUnless):
                jump_if_cond = False
            else:
                raise TypeError('Invalid JumpConditional')
            if not cond ^ jump_if_cond:
                self.program_counter = dest_index
            else:
                self.program_counter += 1
        elif isinstance(instruction, UnaryClassicalInstruction):
            target = instruction.target
            old = self.ram[target.name][target.offset]
            if isinstance(instruction, ClassicalNeg):
                if not isinstance(old, (int, float, np.int_, np.float_)):
                    raise ValueError('NEG requires a data type of REAL or INTEGER; not {}'.format(type(old)))
                self.ram[target.name][target.offset] *= -1
            elif isinstance(instruction, ClassicalNot):
                if not isinstance(old, (bool, np.bool_)):
                    raise ValueError('NOT requires a data type of BIT; not {}'.format(type(old)))
                self.ram[target.name][target.offset] = not old
            else:
                raise TypeError('Invalid UnaryClassicalInstruction')
            self.program_counter += 1
        elif isinstance(instruction, (LogicalBinaryOp, ArithmeticBinaryOp, ClassicalMove)):
            left_ind = instruction.left
            left_val = self.ram[left_ind.name][left_ind.offset]
            if isinstance(instruction.right, MemoryReference):
                right_ind = instruction.right
                right_val = self.ram[right_ind.name][right_ind.offset]
            else:
                right_val = instruction.right
            if isinstance(instruction, ClassicalAnd):
                assert isinstance(left_val, int) and isinstance(right_val, int)
                new_val: Union[int, float] = left_val & right_val
            elif isinstance(instruction, ClassicalInclusiveOr):
                assert isinstance(left_val, int) and isinstance(right_val, int)
                new_val = left_val | right_val
            elif isinstance(instruction, ClassicalExclusiveOr):
                assert isinstance(left_val, int) and isinstance(right_val, int)
                new_val = left_val ^ right_val
            elif isinstance(instruction, ClassicalAdd):
                new_val = left_val + right_val
            elif isinstance(instruction, ClassicalSub):
                new_val = left_val - right_val
            elif isinstance(instruction, ClassicalMul):
                new_val = left_val * right_val
            elif isinstance(instruction, ClassicalDiv):
                new_val = left_val / right_val
            elif isinstance(instruction, ClassicalMove):
                new_val = right_val
            else:
                raise ValueError('Unknown BinaryOp {}'.format(type(instruction)))
            self.ram[left_ind.name][left_ind.offset] = new_val
            self.program_counter += 1
        elif isinstance(instruction, ClassicalExchange):
            left_ind_ex = instruction.left
            right_ind_ex = instruction.right
            tmp = self.ram[left_ind_ex.name][left_ind_ex.offset]
            self.ram[left_ind_ex.name][left_ind_ex.offset] = self.ram[right_ind_ex.name][right_ind_ex.offset]
            self.ram[right_ind_ex.name][right_ind_ex.offset] = tmp
            self.program_counter += 1
        elif isinstance(instruction, Reset):
            self.wf_simulator.reset()
            self.program_counter += 1
        elif isinstance(instruction, ResetQubit):
            raise NotImplementedError('Need to implement in wf simulator')
        elif isinstance(instruction, Wait):
            self.program_counter += 1
        elif isinstance(instruction, Nop):
            self.program_counter += 1
        elif isinstance(instruction, DefGate):
            if instruction.parameters is not None and len(instruction.parameters) > 0:
                raise NotImplementedError('PyQVM does not support parameterized DEFGATEs')
            self.defined_gates[instruction.name] = instruction.matrix
            self.program_counter += 1
        elif isinstance(instruction, RawInstr):
            raise NotImplementedError('PyQVM does not support raw instructions. Parse your program')
        elif isinstance(instruction, Halt):
            return True
        else:
            raise ValueError('Unsupported instruction type: {}'.format(instruction))
        assert self.program is not None
        return self.program_counter == len(self.program)

    def _execute_program(self) -> 'PyQVM':
        self.program_counter = 0
        assert self.program is not None
        halted = len(self.program) == 0
        while not halted:
            halted = self.transition()
        return self

    def execute_once(self, program: Program) -> 'PyQVM':
        """
        Execute one outer loop of a program on the PyQVM without re-initializing its state.

        Note that the PyQVM is stateful. Subsequent calls to :py:func:`execute_once` will not
        automatically reset the wavefunction or the classical RAM. If this is desired,
        consider starting your program with ``RESET``.

        :return: ``self`` to support method chaining.
        """
        self.program = program
        self._extract_defined_gates()
        return self._execute_program()