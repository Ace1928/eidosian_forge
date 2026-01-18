import heapq
from qiskit.circuit.controlledgate import ControlledGate
def _backward_metrics(self, scenario):
    """
        Heuristics to cut the tree in the backward match algorithm.
        Args:
            scenario (MatchingScenarios): scenario for the given match.
        Returns:
            int: length of the match for the given scenario.
        """
    return len(scenario.matches)