from __future__ import annotations
import logging
from collections.abc import Mapping
import numpy as np
import rustworkx as rx
from .types import Swap, Permutation
from .util import PermutationCircuit, permutation_circuit
class ApproximateTokenSwapper:
    """A class for computing approximate solutions to the Token Swapping problem.

    Internally caches the graph and associated datastructures for re-use.
    """

    def __init__(self, graph: rx.PyGraph, seed: int | np.random.Generator | None=None) -> None:
        """Construct an ApproximateTokenSwapping object.

        Args:
            graph: Undirected graph represented a coupling map.
            seed: Seed to use for random trials.
        """
        self.graph = graph
        self.shortest_paths = rx.graph_distance_matrix(graph)
        if isinstance(seed, np.random.Generator):
            self.seed = seed
        else:
            self.seed = np.random.default_rng(seed)

    def distance(self, vertex0: int, vertex1: int) -> int:
        """Compute the distance between two nodes in `graph`."""
        return self.shortest_paths[vertex0, vertex1]

    def permutation_circuit(self, permutation: Permutation, trials: int=4) -> PermutationCircuit:
        """Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Args:
          permutation: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.

        Returns:
          The circuit to implement the permutation
        """
        sequential_swaps = self.map(permutation, trials=trials)
        parallel_swaps = [[swap] for swap in sequential_swaps]
        return permutation_circuit(parallel_swaps)

    def map(self, mapping: Mapping[int, int], trials: int=4, parallel_threshold: int=50) -> list[Swap[int]]:
        """Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Supports partial mappings (i.e. not-permutations) for graphs with missing tokens.

        Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
        ArXiV: https://arxiv.org/abs/1602.05150
        and generalization based on our own work.

        Args:
          mapping: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.
          parallel_threshold: The number of nodes in the graph beyond which the algorithm
                will use parallel processing

        Returns:
          The swaps to implement the mapping
        """
        seed = self.seed.integers(1, 10000)
        return rx.graph_token_swapper(self.graph, mapping, trials, seed, parallel_threshold)