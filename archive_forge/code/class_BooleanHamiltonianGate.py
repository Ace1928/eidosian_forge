import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple
import sympy.parsing.sympy_parser as sympy_parser
import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString
@value.value_equality
class BooleanHamiltonianGate(raw_types.Gate):
    """A gate that represents evolution due to a Hamiltonian from a set of Boolean functions.

    This gate constructs a diagonal gate in the computational basis that encodes in its
    phases classical functions.

    The gate is specified by a list of parameters, $[x_0, x_1, \\dots, x_{n-1}]$, a
    list of boolean expressions that are functions of these parameters,
    $[f_0(x_0,\\dots,x_{n-1}), f_1(x_0,\\dots,x_{n-1}), \\dots f_{p-1}(x_0,\\dots,x_{n-1})]$
    and an angle $t$. For these parameters the gate is

    $$
    \\sum_{x=0}^{2^n-1} e^{i \\frac{t}{2} \\sum_{k=0}^{p-1}f_k(x_0,\\dots,x_{n-1})} |x\\rangle\\langle x|
    $$
    """

    def __init__(self, parameter_names: Sequence[str], boolean_strs: Sequence[str], theta: float):
        """Builds a BooleanHamiltonianGate.

        For each element of a sequence of Boolean expressions, the code first transforms it into a
        polynomial of Pauli Zs that represent that particular expression. Then, we sum all the
        polynomials, thus making a function that goes from a series to Boolean inputs to an integer
        that is the number of Boolean expressions that are true.

        For example, if we were using this gate for the unweighted max-cut problem that is typically
        used to demonstrate the QAOA algorithm, there would be one Boolean expression per edge. Each
        Boolean expression would be true iff the vertices on that are in different cuts (i.e. it's)
        an XOR.

        Then, we compute exp(-j * theta * polynomial), which is unitary because the polynomial is
        Hermitian.

        Args:
            parameter_names: The names of the inputs to the expressions.
            boolean_strs: The list of Sympy-parsable Boolean expressions.
            theta: The evolution time (angle) for the Hamiltonian
        """
        self._parameter_names: Sequence[str] = parameter_names
        self._boolean_strs: Sequence[str] = boolean_strs
        self._theta: float = theta

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,) * len(self._parameter_names)

    def _value_equality_values_(self) -> Any:
        return (self._parameter_names, self._boolean_strs, self._theta)

    def _json_dict_(self) -> Dict[str, Any]:
        return {'parameter_names': self._parameter_names, 'boolean_strs': self._boolean_strs, 'theta': self._theta}

    @classmethod
    def _from_json_dict_(cls, parameter_names, boolean_strs, theta, **kwargs) -> 'cirq.BooleanHamiltonianGate':
        return cls(parameter_names, boolean_strs, theta)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        qubit_map = dict(zip(self._parameter_names, qubits))
        boolean_exprs = [sympy_parser.parse_expr(boolean_str) for boolean_str in self._boolean_strs]
        hamiltonian_polynomial_list = [PauliSum.from_boolean_expression(boolean_expr, qubit_map) for boolean_expr in boolean_exprs]
        return _get_gates_from_hamiltonians(hamiltonian_polynomial_list, qubit_map, self._theta)

    def _has_unitary_(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'cirq.BooleanHamiltonianGate(parameter_names={self._parameter_names!r}, boolean_strs={self._boolean_strs!r}, theta={self._theta!r})'