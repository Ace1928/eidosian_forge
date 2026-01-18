from qiskit.transpiler.basepasses import AnalysisPass
class CountOpsLongestPath(AnalysisPass):
    """Count the operations on the longest path in a :class:`.DAGCircuit`.

    The result is saved in ``property_set['count_ops_longest_path']`` as an integer.
    """

    def run(self, dag):
        """Run the CountOpsLongestPath pass on `dag`."""
        self.property_set['count_ops_longest_path'] = dag.count_ops_longest_path()