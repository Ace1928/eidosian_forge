from __future__ import annotations
import numpy as np
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from .evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
class QAOAAnsatz(EvolvedOperatorAnsatz):
    """A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(self, cost_operator=None, reps: int=1, initial_state: QuantumCircuit | None=None, mixer_operator=None, name: str='QAOA', flatten: bool | None=None):
        """
        Args:
            cost_operator (BaseOperator or OperatorBase, optional): The operator
                representing the cost of the optimization problem, denoted as :math:`U(C, \\gamma)`
                in the original paper. Must be set either in the constructor or via property setter.
            reps (int): The integer parameter p, which determines the depth of the circuit,
                as specified in the original paper, default is 1.
            initial_state (QuantumCircuit, optional): An optional initial state to use.
                If `None` is passed then a set of Hadamard gates is applied as an initial state
                to all qubits.
            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): An optional
                custom mixer to use instead of the global X-rotations, denoted as :math:`U(B, \\beta)`
                in the original paper. Can be an operator or an optionally parameterized quantum
                circuit.
            name (str): A name of the circuit, default 'qaoa'
            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple
                layers of gate objects. By default currently the contents of
                the output circuit will be wrapped in nested objects for
                cleaner visualization. However, if you're using this circuit
                for anything besides visualization its **strongly** recommended
                to set this flag to ``True`` to avoid a large performance
                overhead for parameter binding.
        """
        super().__init__(reps=reps, name=name, flatten=flatten)
        self._cost_operator = None
        self._reps = reps
        self._initial_state: QuantumCircuit | None = initial_state
        self._mixer = mixer_operator
        self._bounds: list[tuple[float | None, float | None]] | None = None
        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool=True) -> bool:
        """Check if the current configuration is valid."""
        valid = True
        if not super()._check_configuration(raise_on_failure):
            return False
        if self.cost_operator is None:
            valid = False
            if raise_on_failure:
                raise ValueError('The operator representing the cost of the optimization problem is not set')
        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError('The number of qubits of the initial state {} does not match the number of qubits of the cost operator {}'.format(self.initial_state.num_qubits, self.num_qubits))
        if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError('The number of qubits of the mixer {} does not match the number of qubits of the cost operator {}'.format(self.mixer_operator.num_qubits, self.num_qubits))
        return valid

    @property
    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
        if self._bounds is not None:
            return self._bounds
        if isinstance(self.mixer_operator, QuantumCircuit):
            return None
        beta_bounds = (0, 2 * np.pi)
        gamma_bounds = (None, None)
        bounds: list[tuple[float | None, float | None]] = []
        if not _is_pauli_identity(self.mixer_operator):
            bounds += self.reps * [beta_bounds]
        if not _is_pauli_identity(self.cost_operator):
            bounds += self.reps * [gamma_bounds]
        return bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: list[tuple[float | None, float | None]] | None) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    @property
    def operators(self) -> list:
        """The operators that are evolved in this circuit.

        Returns:
             List[Union[BaseOperator, OperatorBase, QuantumCircuit]]: The operators to be evolved
                (and circuits) in this ansatz.
        """
        return [self.cost_operator, self.mixer_operator]

    @property
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.

        Returns:
            BaseOperator or OperatorBase: cost operator.
        """
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator.

        Args:
            cost_operator (BaseOperator or OperatorBase, optional): cost operator to set.
        """
        self._cost_operator = cost_operator
        self.qregs = [QuantumRegister(self.num_qubits, name='q')]
        self._invalidate()

    @property
    def reps(self) -> int:
        """Returns the `reps` parameter, which determines the depth of the circuit."""
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        """Sets the `reps` parameter."""
        self._reps = reps
        self._invalidate()

    @property
    def initial_state(self) -> QuantumCircuit | None:
        """Returns an optional initial state as a circuit"""
        if self._initial_state is not None:
            return self._initial_state
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state
        return None

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        """Sets initial state."""
        self._initial_state = initial_state
        self._invalidate()

    @property
    def mixer_operator(self):
        """Returns an optional mixer operator expressed as an operator or a quantum circuit.

        Returns:
            BaseOperator or OperatorBase or QuantumCircuit, optional: mixer operator or circuit.
        """
        if self._mixer is not None:
            return self._mixer
        if self.cost_operator is not None:
            num_qubits = self.cost_operator.num_qubits
            mixer_terms = [('I' * left + 'X' + 'I' * (num_qubits - left - 1), 1) for left in range(num_qubits)]
            mixer = SparsePauliOp.from_list(mixer_terms)
            return mixer
        return None

    @mixer_operator.setter
    def mixer_operator(self, mixer_operator) -> None:
        """Sets mixer operator.

        Args:
            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): mixer
                operator or circuit to set.
        """
        self._mixer = mixer_operator
        self._invalidate()

    @property
    def num_qubits(self) -> int:
        if self._cost_operator is None:
            return 0
        return self._cost_operator.num_qubits

    def _build(self):
        """If not already built, build the circuit."""
        if self._is_built:
            return
        super()._build()
        num_cost = 0 if _is_pauli_identity(self.cost_operator) else 1
        if isinstance(self.mixer_operator, QuantumCircuit):
            num_mixer = self.mixer_operator.num_parameters
        else:
            num_mixer = 0 if _is_pauli_identity(self.mixer_operator) else 1
        betas = ParameterVector('β', self.reps * num_mixer)
        gammas = ParameterVector('γ', self.reps * num_cost)
        reordered = []
        for rep in range(self.reps):
            reordered.extend(gammas[rep * num_cost:(rep + 1) * num_cost])
            reordered.extend(betas[rep * num_mixer:(rep + 1) * num_mixer])
        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)