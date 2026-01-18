import logging
import numpy as np
def _sequential_network(num_qubits: int, links: dict, depth: int) -> np.ndarray:
    """
    Generates a sequential network.

    Args:
        num_qubits: number of qubits.
        links: dictionary of connectivity links.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of ``(2, N)`` that defines layers in qubit network.
    """
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(0, num_qubits - 1):
            for j in range(i + 1, num_qubits):
                if j in links[i]:
                    cnots[0, layer] = i
                    cnots[1, layer] = j
                    layer += 1
                    if layer >= depth:
                        return cnots