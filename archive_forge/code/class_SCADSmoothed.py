import numpy as np
class SCADSmoothed(SCAD):
    """
    The SCAD penalty of Fan and Li, quadratically smoothed around zero.

    This follows Fan and Li 2001 equation (3.7).

    Parameterization follows Boo, Johnson, Li and Tan 2011
    see docstring of SCAD

    Parameters
    ----------
    tau : float
        slope and threshold for linear segment
    c : float
        factor for second threshold
    c0 : float
        threshold for quadratically smoothed segment
    restriction : None or array
        linear constraints for

    Notes
    -----
    TODO: Use delegation instead of subclassing, so smoothing can be added to
    all penalty classes.
    """

    def __init__(self, tau, c=3.7, c0=None, weights=1.0, restriction=None):
        super().__init__(tau, c=c, weights=weights)
        self.tau = tau
        self.c = c
        self.c0 = c0 if c0 is not None else tau * 0.1
        if self.c0 > tau:
            raise ValueError('c0 cannot be larger than tau')
        c0 = self.c0
        weights = self.weights
        self.weights = 1.0
        deriv_c0 = super().deriv(c0)
        value_c0 = super().func(c0)
        self.weights = weights
        self.aq1 = value_c0 - 0.5 * deriv_c0 * c0
        self.aq2 = 0.5 * deriv_c0 / c0
        self.restriction = restriction

    def func(self, params):
        weights = self._null_weights(params)
        if self.restriction is not None and np.size(params) > 1:
            params = self.restriction.dot(params)
        self_weights = self.weights
        self.weights = 1.0
        value = super().func(params[None, ...])
        self.weights = self_weights
        value -= self.aq1
        p_abs = np.atleast_1d(np.abs(params))
        mask = p_abs < self.c0
        p_abs_masked = p_abs[mask]
        value[mask] = self.aq2 * p_abs_masked ** 2
        return (weights * value).sum(0)

    def deriv(self, params):
        weights = self._null_weights(params)
        if self.restriction is not None and np.size(params) > 1:
            params = self.restriction.dot(params)
        self_weights = self.weights
        self.weights = 1.0
        value = super().deriv(params)
        self.weights = self_weights
        p = np.atleast_1d(params)
        mask = np.abs(p) < self.c0
        value[mask] = 2 * self.aq2 * p[mask]
        if self.restriction is not None and np.size(params) > 1:
            return weights * value.dot(self.restriction)
        else:
            return weights * value

    def deriv2(self, params):
        weights = self._null_weights(params)
        if self.restriction is not None and np.size(params) > 1:
            params = self.restriction.dot(params)
        self_weights = self.weights
        self.weights = 1.0
        value = super().deriv2(params)
        self.weights = self_weights
        p = np.atleast_1d(params)
        mask = np.abs(p) < self.c0
        value[mask] = 2 * self.aq2
        if self.restriction is not None and np.size(params) > 1:
            return (self.restriction.T * (weights * value)).dot(self.restriction)
        else:
            return weights * value