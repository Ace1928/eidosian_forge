from qiskit.dagcircuit.dagdependency import DAGDependency
Build a ``DAGDependency`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    