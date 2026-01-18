import numpy as np
import pennylane as qml
from pennylane.utils import _flatten, unflatten
def _find_optimal_generators(self, objective_fn, x, generators, d):
    """Optimizer for the generators.

        Optimizes for the best generator at position ``d``.

        Args:
            objective_fn (function): The objective function for optimization. It must have the
                signature ``objective_fn(x, generators=None)`` with a sequence of the values ``x``
                and a list of the gates ``generators`` as inputs, returning a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized over or a single float with the initial value
            generators (list[~.Operation]): list containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over
            d (int): the position in the input sequence ``x`` containing the value to be optimized

        Returns:
            tuple: tuple containing the parameter value and generator that, at position ``d`` in
            ``x`` and ``generators``, optimizes the objective function
        """
    params_opt_d = x[d]
    generators_opt_d = generators[d]
    params_opt_cost = objective_fn(x, generators)
    for generator in self.possible_generators:
        generators[d] = generator
        x = self._rotosolve(objective_fn, x, generators, d)
        params_cost = objective_fn(x, generators)
        if params_cost <= params_opt_cost:
            params_opt_d = x[d]
            params_opt_cost = params_cost
            generators_opt_d = generator
    return (params_opt_d, generators_opt_d)