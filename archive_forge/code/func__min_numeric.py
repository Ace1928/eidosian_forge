from inspect import signature
import numpy as np
from scipy.optimize import brute, shgo
import pennylane as qml
def _min_numeric(self, objective_fn, spectrum):
    """Numerically minimize a trigonometric function that depends on a
        single parameter. Uses potentially large numbers of function evaluations,
        depending on the used substep_optimizer. The optimization method and
        options are stored in ``RotosolveOptimizer.substep_optimizer``
        and ``RotosolveOptimizer.substep_kwargs``.

        Args:
            objective_fn (callable): Trigonometric function to minimize

        Returns:
            float: Position of the minimum of ``objective_fn``
            float: Value of the minimum of ``objective_fn``

        The returned position is guaranteed to lie within :math:`(-\\pi, \\pi]`.
        """
    opt_kwargs = self.substep_kwargs.copy()
    if 'bounds' not in self.substep_kwargs:
        spectrum = qml.math.array(spectrum)
        half_width = np.pi / qml.math.min(spectrum[spectrum > 0])
        opt_kwargs['bounds'] = ((-half_width, half_width),)
    x_min, y_min = self.substep_optimizer(objective_fn, **opt_kwargs)
    if y_min is None:
        y_min = objective_fn(x_min)
    return (x_min, y_min)