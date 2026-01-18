from functools import partial
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
from qiskit.quantum_info.operators import Clifford
CollectCliffords initializer.

        Args:
            do_commutative_analysis (bool): if True, exploits commutativity relations
                between nodes.
            split_blocks (bool): if True, splits collected blocks into sub-blocks
                over disjoint qubit subsets.
            min_block_size (int): specifies the minimum number of gates in the block
                for the block to be collected.
            split_layers (bool): if True, splits collected blocks into sub-blocks
                over disjoint qubit subsets.
            collect_from_back (bool): specifies if blocks should be collected started
                from the end of the circuit.
        