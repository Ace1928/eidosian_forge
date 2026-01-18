from os.path import basename, isfile
from typing import Callable, Optional
from qiskit.circuit import QuantumCircuit
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from .classical_element import ClassicalElement
@HAS_TWEEDLEDUM.require_in_instance
class BooleanExpression(ClassicalElement):
    """The Boolean Expression gate."""

    def __init__(self, expression: str, name: str=None, var_order: list=None) -> None:
        """
        Args:
            expression (str): The logical expression string.
            name (str): Optional. Instruction gate name. Otherwise part of the expression is
               going to be used.
            var_order(list): A list with the order in which variables will be created.
               (default: by appearance)
        """
        from tweedledum import BoolFunction
        self._tweedledum_bool_expression = BoolFunction.from_expression(expression, var_order=var_order)
        short_expr_for_name = expression[:10] + '...' if len(expression) > 13 else expression
        num_qubits = self._tweedledum_bool_expression.num_outputs() + self._tweedledum_bool_expression.num_inputs()
        super().__init__(name or short_expr_for_name, num_qubits=num_qubits, params=[])

    def simulate(self, bitstring: str) -> bool:
        """Evaluate the expression on a bitstring.

        This evaluation is done classically.

        Args:
            bitstring: The bitstring for which to evaluate.

        Returns:
            bool: result of the evaluation.
        """
        from tweedledum import BitVec
        bits = []
        for bit in bitstring:
            bits.append(BitVec(1, bit))
        return bool(self._tweedledum_bool_expression.simulate(*bits))

    def synth(self, registerless: bool=True, synthesizer: Optional[Callable[['BooleanExpression'], QuantumCircuit]]=None):
        """Synthesis the logic network into a :class:`~qiskit.circuit.QuantumCircuit`.

        Args:
            registerless: Default ``True``. If ``False`` uses the parameter names
                to create registers with those names. Otherwise, creates a circuit with a flat
                quantum register.
            synthesizer: A callable that takes self and returns a Tweedledum
                circuit.
        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        """
        if registerless:
            qregs = None
        else:
            qregs = None
        if synthesizer is None:
            from .utils import tweedledum2qiskit
            from tweedledum.synthesis import pkrm_synth
            truth_table = self._tweedledum_bool_expression.truth_table(output_bit=0)
            return tweedledum2qiskit(pkrm_synth(truth_table), name=self.name, qregs=qregs)
        return synthesizer(self)

    def _define(self):
        """The definition of the boolean expression is its synthesis"""
        self.definition = self.synth()

    @classmethod
    def from_dimacs_file(cls, filename: str):
        """Create a BooleanExpression from the string in the DIMACS format.
        Args:
            filename: A file in DIMACS format.

        Returns:
            BooleanExpression: A gate for the input string

        Raises:
            FileNotFoundError: If filename is not found.
        """
        HAS_TWEEDLEDUM.require_now('BooleanExpression')
        from tweedledum import BoolFunction
        expr_obj = cls.__new__(cls)
        if not isfile(filename):
            raise FileNotFoundError('The file %s does not exists.' % filename)
        expr_obj._tweedledum_bool_expression = BoolFunction.from_dimacs_file(filename)
        num_qubits = expr_obj._tweedledum_bool_expression.num_inputs() + expr_obj._tweedledum_bool_expression.num_outputs()
        super(BooleanExpression, expr_obj).__init__(name=basename(filename), num_qubits=num_qubits, params=[])
        return expr_obj