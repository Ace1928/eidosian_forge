import logging
import numpy as np

    Generates a line based CNOT structure.

    Args:
        num_qubits: number of qubits.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of size ``2 x L`` that defines layers in qubit network.
    