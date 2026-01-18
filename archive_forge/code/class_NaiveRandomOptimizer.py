import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
class NaiveRandomOptimizer(oe.path_random.RandomOptimizer):

    @staticmethod
    def random_path(r, n, inputs, output, size_dict):
        """Picks a completely random contraction order.
            """
        np.random.seed(r)
        ssa_path = []
        remaining = set(range(n))
        while len(remaining) > 1:
            i, j = np.random.choice(list(remaining), size=2, replace=False)
            remaining.add(n + len(ssa_path))
            remaining.remove(i)
            remaining.remove(j)
            ssa_path.append((i, j))
        cost, size = oe.path_random.ssa_path_compute_cost(ssa_path, inputs, output, size_dict)
        return (ssa_path, cost, size)

    def setup(self, inputs, output, size_dict):
        self.was_used = True
        n = len(inputs)
        trial_fn = self.random_path
        trial_args = (n, inputs, output, size_dict)
        return (trial_fn, trial_args)