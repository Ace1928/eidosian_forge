from qiskit.transpiler.basepasses import AnalysisPass
class ContainsInstruction(AnalysisPass):
    """An analysis pass to detect if the DAG contains a specific instruction.

    This pass takes in a single instruction name for example ``'delay'`` and
    will set the property set ``contains_delay`` to ``True`` if the DAG contains
    that instruction and ``False`` if it does not.
    """

    def __init__(self, instruction_name, recurse: bool=True):
        """ContainsInstruction initializer.

        Args:
            instruction_name (str | Iterable[str]): The instruction or instructions to check are in
                the DAG. The output in the property set is set to ``contains_`` prefixed on each
                value for this parameter.
            recurse (bool): if ``True`` (default), then recurse into control-flow operations.
        """
        super().__init__()
        self._instruction_names = {instruction_name} if isinstance(instruction_name, str) else set(instruction_name)
        self._recurse = recurse

    def run(self, dag):
        """Run the ContainsInstruction pass on dag."""
        names = dag.count_ops(recurse=self._recurse)
        for name in self._instruction_names:
            self.property_set[f'contains_{name}'] = name in names