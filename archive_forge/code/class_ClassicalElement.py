from abc import ABCMeta, abstractmethod
from qiskit.circuit import Gate
class ClassicalElement(Gate, metaclass=ABCMeta):
    """The classical element gate."""

    @abstractmethod
    def simulate(self, bitstring: str) -> bool:
        """Evaluate the expression on a bitstring.

        This evaluation is done classically.

        Args:
            bitstring: The bitstring for which to evaluate.

        Returns:
            bool: result of the evaluation.
        """
        pass

    @abstractmethod
    def synth(self, registerless=True, synthesizer=None):
        """Synthesis the logic network into a :class:`~qiskit.circuit.QuantumCircuit`.

        Args:
            registerless (bool): Default ``True``. If ``False`` uses the parameter names
                to create registers with those names. Otherwise, creates a circuit with a flat
                quantum register.
            synthesizer (callable): A callable that takes a Logic Network and returns a Tweedledum
                circuit.
        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        """
        pass

    def _define(self):
        """The definition of the boolean expression is its synthesis"""
        self.definition = self.synth()