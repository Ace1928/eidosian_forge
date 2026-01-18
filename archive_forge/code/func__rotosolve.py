import numpy as np
import pennylane as qml
from pennylane.utils import _flatten, unflatten
@staticmethod
def _rotosolve(objective_fn, x, generators, d):
    """The rotosolve step for one parameter and one set of generators.

        Updates the parameter :math:`\\theta_d` based on Equation 1 in
        `Ostaszewski et al. (2021) <https://doi.org/10.22331/q-2021-01-28-391>`_.

        Args:
            objective_fn (function): The objective function for optimization. It must have the
                signature ``objective_fn(x, generators=None)`` with a sequence of the values ``x``
                and a list of the gates ``generators`` as inputs, returning a single value.
            x (Union[Sequence[float], float]): sequence containing the initial values of the
                variables to be optimized overs or a single float with the initial value
            generators (list[~.Operation]): list containing the initial ``pennylane.ops.qubit``
                operators to be used in the circuit and optimized over
            d (int): the position in the input sequence ``x`` containing the value to be optimized

        Returns:
            array: the input sequence ``x`` with the value at position ``d`` optimized
        """

    def insert(x, d, theta):
        x[d] = theta
        return x
    H_0 = float(objective_fn(insert(x, d, 0), generators))
    H_p = float(objective_fn(insert(x, d, np.pi / 2), generators))
    H_m = float(objective_fn(insert(x, d, -np.pi / 2), generators))
    a = np.arctan2(2 * H_0 - H_p - H_m, H_p - H_m)
    x[d] = -np.pi / 2 - a
    if x[d] <= -np.pi:
        x[d] += 2 * np.pi
    return x